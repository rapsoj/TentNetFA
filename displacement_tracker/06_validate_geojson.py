import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import click
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features, mask
from shapely.geometry import box


@dataclass(frozen=True)
class RasterMetadata:
    """Stores the necessary metadata for the output difference raster."""

    transform: rasterio.Affine
    width: int
    height: int
    crs: rasterio.crs.CRS


def extract_date_from_path(path: str) -> Optional[datetime]:
    """Matches YYYY-MM-DD or YYYYMMDD in filenames."""
    match = re.search(r"(\d{4}-\d{2}-\d{2})|(\d{8})", os.path.basename(path))
    if not match:
        return None
    date_str = match.group(0).replace("-", "")
    return datetime.strptime(date_str, "%Y%m%d")


def get_point_counts(
    gdf: gpd.GeoDataFrame, out_shape: Tuple[int, int], transform: rasterio.Affine
) -> np.ndarray:
    """Rasterizes point geometries into counts per cell."""
    if gdf.empty:
        return np.zeros(out_shape, dtype=np.float32)

    # Create (geometry, 1) pairs. rasterize expects geometries in the target CRS.
    shapes = ((geom, 1) for geom in gdf.geometry)

    return features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        merge_alg=rasterio.enums.MergeAlg.add,
        fill=0,
        dtype="float32",
    )


@click.command()
@click.option("--pred-dir", type=click.Path(exists=True), required=True)
@click.option("--val-dir", type=click.Path(exists=True), required=True)
@click.option("--master-grid", type=click.Path(exists=True), required=True)
@click.option("--out-dir", type=click.Path(), default="diff_results")
def cli(pred_dir: str, val_dir: str, master_grid: str, out_dir: str):
    """
    Validates point predictions against ground truth by differencing
    rasterized counts within a master grid framework.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # validation files and dates
    val_paths = [
        os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".gpkg")
    ]
    val_map = {
        extract_date_from_path(p): p for p in val_paths if extract_date_from_path(p)
    }

    if not val_map:
        raise click.ClickException("No date-stamped validation files found.")

    with rasterio.open(master_grid) as src_grid:
        # src_grid = rasterio.open(master_grid)

        raster_crs = src_grid.crs

        pred_files = [f for f in os.listdir(pred_dir) if f.endswith(".geojson")]
        for pred_file in pred_files:
            # pred_file = pred_files[0]

            # prediction filepath and date
            pred_path = os.path.join(pred_dir, pred_file)
            pred_date = extract_date_from_path(pred_path)

            if not pred_date:
                continue

            # temporal matching to validation data
            val_dates = list(val_map.keys())
            closest_date = min(val_dates, key=lambda d: abs(d - pred_date))

            # load and project prediction and validation data
            pred_gdf = gpd.read_file(pred_path).to_crs(raster_crs)
            val_gdf = gpd.read_file(val_map[closest_date]).to_crs(raster_crs)

            if pred_gdf.empty:
                continue

            # analysis mask (convex hull around predicted tents)
            prediction_extent_geom = pred_gdf.union_all().convex_hull

            # apply mask to mastergrid
            try:
                out_image, out_transform = mask.mask(
                    src_grid,
                    [prediction_extent_geom],
                    crop=True,
                    nodata=np.nan,
                )
            except ValueError:
                click.echo(f"Skipping {pred_file}: No overlap.")
                continue

            # extract dimensions from the masked array
            # out_image shape is (bands, rows, cols)
            grid_shape = (out_image.shape[1], out_image.shape[2])

            # rasterize predictions
            pred_raster = get_point_counts(pred_gdf, grid_shape, out_transform)

            # clip validation points to the analysis mask
            val_in_hull = val_gdf.clip(prediction_extent_geom)

            # rasterise validation points
            val_raster = get_point_counts(val_in_hull, grid_shape, out_transform)

            # --- calculate prediction "error" ---#
            diff = pred_raster - val_raster

            # mask as an array for rasters (outside the mask is False (0))
            mask_array = features.geometry_mask(
                [prediction_extent_geom],
                out_shape=grid_shape,
                transform=out_transform,
                invert=True,
            )

            # no data value to apply outside the mask
            nodata_val = -9999.0

            # mask all three arrays
            for r in [pred_raster, val_raster, diff]:
                r[~mask_array] = nodata_val

            # save three rasters (predicted count, validation count, and difference)
            meta = src_grid.meta.copy()
            meta.update(
                {
                    "height": grid_shape[0],
                    "width": grid_shape[1],
                    "transform": out_transform,
                    "dtype": "float32",
                    "nodata": nodata_val,
                    "count": 1,
                }
            )

            base_name = pred_file.replace(".geojson", "")
            outputs = {"diff": diff, "pred_count": pred_raster, "val_count": val_raster}

            for suffix, data in outputs.items():
                out_path = os.path.join(out_dir, f"{base_name}_{suffix}.tif")
                with rasterio.open(out_path, "w", **meta) as dest:
                    dest.write(data, 1)

            click.echo(f"Exported set for: {base_name}")


if __name__ == "__main__":
    cli()
