"""Elevation derivatives: gradient, slope, aspect (azimuth), surface normals.

Horn's (1981) 3x3 finite-difference operator is used; it is the standard in
GIS (GDAL, ArcGIS) and is less noise sensitive than central differences
because it averages over the 8-neighbourhood with weights 1-2-1.

Conventions (raster rows grow southwards, columns eastwards):
  dz/dx  - east derivative
  dz/dy  - north derivative
  slope  = atan(|∇z|)                     in [0, 90]
  aspect = compass azimuth of steepest *descent*, clockwise from north,
           atan2(-dz/dx, -dz/dy) mod 360  (0=N, 90=E, 180=S, 270=W)
  normal = (-dz/dx, -dz/dy, 1) / ||·||   (upward unit normal, ENU frame)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
_KY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], float)  # +north (row 0 is north)


def horn_gradient(z: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (dz/dx_east, dz/dy_north) using Horn's operator."""
    # ndi.correlate keeps kernel orientation (no flip)
    gx = ndi.correlate(z, _KX, mode="nearest") / (8.0 * res)
    gy = ndi.correlate(z, _KY, mode="nearest") / (8.0 * res)
    return gx, gy


def slope_aspect(z: np.ndarray, res: float):
    gx, gy = horn_gradient(z, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = (np.degrees(np.arctan2(-gx, -gy)) + 360.0) % 360.0
    aspect[np.hypot(gx, gy) < 1e-6] = np.nan  # undefined on flat cells
    return slope, aspect, gx, gy


def normals_from_gradient(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def masked_gaussian(z: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: smooth z using only pixels in mask (no bleeding from ground)."""
    m = mask.astype(float)
    num = ndi.gaussian_filter(np.where(mask, z, 0.0), sigma)
    den = ndi.gaussian_filter(m, sigma)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), z)
    return np.where(mask, out, z)


def plane_slope_aspect(a: float, b: float) -> tuple[float, float]:
    """Slope/aspect of plane z = a*x_east + b*y_north + c."""
    slope = float(np.degrees(np.arctan(np.hypot(a, b))))
    aspect = float((np.degrees(np.arctan2(-a, -b)) + 360.0) % 360.0)
    return slope, aspect
