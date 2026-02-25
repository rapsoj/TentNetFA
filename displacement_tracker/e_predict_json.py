import yaml
import click
import torch
import numpy as np
import json
from pathlib import Path
from scipy.ndimage import label, center_of_mass

from displacement_tracker.paired_image_dataset import PairedImageDataset
from displacement_tracker.simple_cnn import SimpleCNN
from displacement_tracker.util.logging_config import setup_logging
from displacement_tracker.util.distance import interpolate_centroid
from displacement_tracker.util.deduplication import merge_close_points_global
from displacement_tracker.util.tiff_predictions import (
    save_prediction_tiff,
    merge_prediction_tiffs,
)

LOGGER = setup_logging("predict_json")


def predict(
    dataset, model, device, processing_cfg, sample_cfg=None, validation_tifs=False
):
    """
    Run predictions and extract centroids of labeled regions above min_area for each tile.
    (No intra-tile spatial deduplication here; deduplication is performed globally after prediction.)
    """
    threshold = processing_cfg.get("threshold", 0.5)
    min_area = processing_cfg.get("min_area", 20)
    crop_pixels = processing_cfg.get("crop_pixels", 0)
    agreement = processing_cfg.get("agreement", False)
    min_distance_m = processing_cfg.get("min_distance_m", 2.0)

    if sample_cfg and sample_cfg.get("enable", True):
        total = len(dataset)
        size = min(sample_cfg.get("size", total), total)
        seed = sample_cfg.get("seed", None)
        frac = float(size) / float(total)
        subsets, _ = dataset.create_subsets([frac, 1 - frac], seed=seed)
        subset = subsets[0]
        LOGGER.info(f"🔹 Using random sample of {size}/{total} tiles for prediction.")
    else:
        subset = dataset
        LOGGER.info(f"🔹 Using all {len(dataset)} tiles for prediction.")

    results = []
    with torch.no_grad():
        for i, entry in enumerate(subset):
            try:
                LOGGER.info(f"Predicting image {i + 1}/{len(subset)}")
                diff = entry["feature"] - entry["prewar"]
                feats = torch.cat((entry["feature"], entry["prewar"], diff), dim=0)
                feats = feats.unsqueeze(0).to(device)  # add batch dim
                outputs = model(feats)
                probs_np = outputs.cpu().squeeze().numpy()

                bounds = json.loads(entry["meta"])

                if crop_pixels > 0:
                    probs_np[:crop_pixels, :] = 0
                    probs_np[-crop_pixels:, :] = 0
                    probs_np[:, :crop_pixels] = 0
                    probs_np[:, -crop_pixels:] = 0

                if validation_tifs:
                    tiff_dir = Path(
                        processing_cfg.get("tiff_output_dir", "prediction_tiffs")
                    )
                    tiff_dir.mkdir(parents=True, exist_ok=True)

                    origin = bounds.get("origin_image", f"tile_{i}")
                    tiff_path = tiff_dir / f"{Path(origin).stem}_{i}_pred.tif"

                    save_prediction_tiff(probs_np, bounds, tiff_path)

                # Threshold
                mask = probs_np > threshold
                # Label regions
                labeled, num_features = label(mask)
                # Filter by min_area and extract centroids
                coords = []
                shape = mask.shape
                for region_id in range(1, num_features + 1):
                    region_mask = labeled == region_id
                    area = np.sum(region_mask)
                    if area > min_area:
                        centroid = center_of_mass(region_mask)
                        try:
                            coord = interpolate_centroid(centroid, bounds, shape)
                            # no intra-tile duplicate removal here; do global merge later
                            coords.append(coord)
                        except Exception as exc:
                            LOGGER.warning(f"Interpolation error: {exc}")
                results.append({"bounds": bounds, "coordinates": coords})
                LOGGER.info(f"Found {len(coords)} tents (pre-merge).")
            except Exception as exc:
                LOGGER.warning(f"Prediction error: {exc}")

    flat_results = [pt for res in results for pt in res["coordinates"]]
    LOGGER.info(f"Total number of tents (pre-merge): {len(flat_results)}")

    # Only keep points with agreement between overlaps
    if isinstance(
        agreement, bool
    ):  # convert to int, False -> 1 point agreement, True -> 2 point agreement
        agreement = 2 if agreement else 1

    # global deduplication in meters
    merged_coords = merge_close_points_global(
        flat_results, min_distance_m=min_distance_m, agreement=agreement
    )
    LOGGER.info(f"Total number of tents (post-merge): {len(merged_coords)}")
    return merged_coords


def save_geojson(points, out_path):
    """
    Save results to a GeoJSON file.
    If deduplicated_points is provided, ONLY write those cleaned points (no original raw points,
    no deduplicated flag). Otherwise, fall back to writing per-tile polygons and points.
    """
    features = []

    # Write only cleaned/deduplicated points (one feature per merged centroid)
    for lat, lon in points:
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"name": "tents"},
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    LOGGER.info(f"GeoJSON saved to {out_path}")


@click.command()
@click.argument("config", type=click.Path(exists=True))
def cli(config) -> None:
    with open(config, "r") as f:
        params = yaml.safe_load(f)

    if "prediction" not in params:
        raise click.ClickException("Missing required config key: prediction")

    pred_cfg = params["prediction"]
    sample_cfg = pred_cfg.get("sample", {})
    out_path = pred_cfg.get("output", "predictions.geojson")
    processing_cfg = pred_cfg.get("processing", {})
    device = pred_cfg.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PairedImageDataset(pred_cfg["input"])
    model = SimpleCNN.from_pth(
        pred_cfg["model"], model_args={"n_channels": 3, "n_classes": 1}
    )
    validation_tifs = pred_cfg.get("validation_tifs", False)

    device = torch.device(device)
    model.to(device)

    # run predictions (per-tile centroids, no intra-tile dedupe)
    results = predict(
        ds, model, device, processing_cfg, sample_cfg, validation_tifs=validation_tifs
    )
    # Save only the cleaned/deduplicated points (no raw points, no deduplicated flag)
    save_geojson(results, out_path)

    if validation_tifs:
        tiff_dir = Path(processing_cfg.get("tiff_output_dir", "prediction_tiffs"))
        mosaic_out = Path(pred_cfg.get("tiff_mosaic_output", "predictions_mosaic.tif"))
        merge_prediction_tiffs(tiff_dir, str(mosaic_out))


if __name__ == "__main__":
    cli()
