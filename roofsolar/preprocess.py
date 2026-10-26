"""Noise regularisation and missing-data reconstruction for the DSM.

Two separate problems are handled:

1. *Outliers* (spikes / pits typical of photogrammetric or LiDAR DSMs) are
   detected with a robust local test - the deviation from a local median,
   compared against the local median absolute deviation (MAD).  Outliers are
   turned into missing data rather than "corrected", so that one code path
   deals with all bad pixels.

2. *Missing data* (nodata, NaN, removed outliers) is reconstructed by
   harmonic inpainting: we solve the discrete Laplace equation
   ``∇²z = 0`` on the missing pixels with the valid neighbours as Dirichlet
   boundary conditions.  Harmonic interpolation is the minimum-energy
   (membrane) surface, has no overshoot (maximum principle), and is exact for
   planar data - which is what roofs mostly are.  After plane fitting, any
   hole that lies inside a roof face is replaced by the face plane itself
   (see ``segmentation.regularized_dsm``), which is the geometrically correct
   reconstruction for a planar facet.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


def detect_outliers(z: np.ndarray, window=5, k_mad=6.0, min_abs=0.5) -> np.ndarray:
    """Boolean mask of spikes/pits relative to a robust local model."""
    zf = np.where(np.isfinite(z), z, np.nanmedian(z))
    med = ndi.median_filter(zf, size=window)
    dev = np.abs(zf - med)
    mad = ndi.median_filter(dev, size=window) * 1.4826
    thr = np.maximum(k_mad * mad, min_abs)
    return np.isfinite(z) & (dev > thr) & _isolated(dev > thr)


def _isolated(mask: np.ndarray, max_area_px=25) -> np.ndarray:
    """Only small blobs count as outliers; large coherent deviations are real structures."""
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes <= max_area_px
    return keep[lab]


def harmonic_inpaint(z: np.ndarray, missing: np.ndarray) -> np.ndarray:
    """Fill ``missing`` pixels by solving Laplace's equation (sparse direct solve)."""
    out = z.copy()
    missing = missing.copy()
    if not missing.any():
        return out
    h, w = z.shape
    idx = -np.ones((h, w), np.int64)
    rr, cc = np.nonzero(missing)
    idx[rr, cc] = np.arange(rr.size)
    n = rr.size
    A = lil_matrix((n, n))
    b = np.zeros(n)
    for k, (r, c) in enumerate(zip(rr, cc)):
        deg = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r2, c2 = r + dr, c + dc
            if 0 <= r2 < h and 0 <= c2 < w:
                deg += 1
                j = idx[r2, c2]
                if j >= 0:
                    A[k, j] = -1.0
                else:
                    b[k] += out[r2, c2]
        A[k, k] = deg
    sol = spsolve(csr_matrix(A), b)
    out[rr, cc] = sol
    return out


def preprocess_dsm(dsm: np.ndarray, cfg) -> tuple[np.ndarray, dict]:
    """Return a gap-free, de-spiked DSM and a report of what was changed."""
    missing = ~np.isfinite(dsm)
    outliers = detect_outliers(dsm, cfg.outlier_window, cfg.outlier_k_mad, cfg.outlier_min_abs)
    bad = missing | outliers
    filled = harmonic_inpaint(np.where(bad, 0.0, dsm), bad) if bad.any() else dsm.copy()
    smooth = ndi.median_filter(filled, size=cfg.median_size) if cfg.median_size > 1 else filled
    report = {
        "missing_px": int(missing.sum()),
        "outlier_px": int(outliers.sum()),
        "filled_px": int(bad.sum()),
    }
    return smooth, {"report": report, "missing": missing, "outliers": outliers, "filled": filled}
