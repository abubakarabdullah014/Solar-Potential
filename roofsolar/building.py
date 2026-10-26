"""Ground model (DTM), normalised DSM and building footprint extraction.

* DTM: grey-scale morphological opening with a disk larger than half the
  building width removes every elevated object (an opening keeps only
  structures wider than the structuring element).  It is computed on a
  decimated grid for speed and smoothed afterwards.
* nDSM = DSM - DTM, i.e. object height above ground.
* Vegetation is separated from roofs with the Excess-Green index
  ExG = (2G - R - B)/(R + G + B) from the RGB orthophoto; the threshold is
  chosen automatically with Otsu's method on the elevated pixels.
* The building is the largest connected elevated, non-vegetated component
  that overlaps the parcel (area of interest given by the clipped DSM).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.morphology import disk


def estimate_dtm(dsm: np.ndarray, res: float, radius: float, down: int = 4) -> np.ndarray:
    small = dsm[::down, ::down]
    r_px = max(1, int(round(radius / (res * down))))
    fp = disk(r_px).astype(bool)
    opened = ndi.grey_opening(small, footprint=fp, mode="nearest")
    opened = ndi.uniform_filter(opened, size=max(3, r_px // 2), mode="nearest")
    zoom = (dsm.shape[0] / small.shape[0], dsm.shape[1] / small.shape[1])
    dtm = ndi.zoom(opened, zoom, order=1, mode="nearest")[: dsm.shape[0], : dsm.shape[1]]
    return np.minimum(dtm, dsm)


def excess_green(rgb: np.ndarray) -> np.ndarray:
    r, g, b = [rgb[..., i].astype(float) for i in range(3)]
    return (2 * g - r - b) / (r + g + b + 1e-6)


def extract_building(dsm, rgb, parcel, res, cfg):
    dtm = estimate_dtm(dsm, res, cfg.dtm_radius, cfg.dtm_downsample)
    ndsm = dsm - dtm
    elevated = ndsm > cfg.min_height
    exg = excess_green(rgb)
    thr = cfg.exg_threshold
    if thr is None:
        thr = float(threshold_otsu(exg[elevated])) if elevated.sum() > 100 else 0.05
    veg = exg > thr
    cand = elevated & ~veg
    r = max(1, int(round(cfg.opening_radius / res)))
    cand = ndi.binary_opening(cand, structure=disk(r))
    lab, n = ndi.label(cand)
    if n == 0:
        raise RuntimeError("No elevated non-vegetated object found")
    idx = np.arange(1, n + 1)
    area = ndi.sum(np.ones_like(lab), lab, idx) * res * res
    overlap = ndi.sum(parcel, lab, idx) * res * res
    score = np.where(area >= cfg.min_building_area, overlap, -1)
    best = int(idx[np.argmax(score)])
    mask = lab == best
    mask = ndi.binary_fill_holes(mask)
    mask = ndi.binary_closing(mask, structure=disk(2)) | mask
    mask = ndi.binary_fill_holes(mask)
    info = {"exg_threshold": thr, "area_m2": float(mask.sum() * res * res),
            "ground_height": float(np.median(dtm[mask]))}
    return mask, dtm, ndsm, veg, info
