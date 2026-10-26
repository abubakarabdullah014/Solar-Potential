"""Raster I/O and co-registration onto one common grid."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform as warp_transform


@dataclass
class Scene:
    dsm: np.ndarray          # float64, NaN = missing
    rgb: np.ndarray          # uint8 HxWx3
    parcel: np.ndarray       # bool, area of interest (valid pixels of the clipped DSM)
    transform: rasterio.Affine
    crs: rasterio.crs.CRS
    res: float

    @property
    def shape(self):
        return self.dsm.shape

    def pixel_to_world(self, rows, cols):
        """Pixel centres -> map coordinates (x east, y north)."""
        rows = np.asarray(rows, float)
        cols = np.asarray(cols, float)
        x = self.transform.c + (cols + 0.5) * self.transform.a
        y = self.transform.f + (rows + 0.5) * self.transform.e
        return x, y

    def lonlat_center(self):
        h, w = self.shape
        x, y = self.pixel_to_world(h / 2, w / 2)
        lon, lat = warp_transform(self.crs, "EPSG:4326", [float(x)], [float(y)])
        return lon[0], lat[0]


def _read_float(path):
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float64)
        if ds.nodata is not None:
            a[a == ds.nodata] = np.nan
        a[~np.isfinite(a)] = np.nan
        return a, ds.transform, ds.crs


def load_scene(dsm_full: str, rgb: str, dsm_clipped: str | None = None) -> Scene:
    """Load the full DSM as reference grid; resample RGB and parcel mask onto it.

    The clipped DSM is used only as an area-of-interest mask (its valid pixels),
    because its heights are identical to the full DSM where both exist.
    """
    dsm, tr, crs = _read_float(dsm_full)
    h, w = dsm.shape

    with rasterio.open(rgb) as ds:
        if ds.transform == tr and ds.shape == (h, w):
            img = ds.read([1, 2, 3])
        else:
            img = np.zeros((3, h, w), np.uint8)
            for b in range(3):
                reproject(ds.read(b + 1), img[b], src_transform=ds.transform, src_crs=ds.crs,
                          dst_transform=tr, dst_crs=crs, resampling=Resampling.bilinear)
    img = np.moveaxis(img, 0, -1).copy()

    if dsm_clipped:
        with rasterio.open(dsm_clipped) as ds:
            src = ds.read(1).astype(np.float64)
            valid = np.ones_like(src, np.uint8)
            if ds.nodata is not None:
                valid[src == ds.nodata] = 0
            valid[~np.isfinite(src)] = 0
            parcel = np.zeros((h, w), np.uint8)
            reproject(valid, parcel, src_transform=ds.transform, src_crs=ds.crs,
                      dst_transform=tr, dst_crs=crs, resampling=Resampling.nearest)
            parcel = parcel.astype(bool)
    else:
        parcel = np.ones((h, w), bool)

    return Scene(dsm=dsm, rgb=img, parcel=parcel, transform=tr, crs=crs, res=abs(tr.a))


def write_geotiff(path, array, scene: Scene, nodata=None, dtype=None):
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None]
    dtype = dtype or arr.dtype
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=dtype, crs=scene.crs, transform=scene.transform,
                       nodata=nodata, compress="deflate") as ds:
        ds.write(arr.astype(dtype))
