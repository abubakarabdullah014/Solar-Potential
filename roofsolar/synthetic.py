"""Synthetic DSM with known ground truth (hip roof + chimney) for testing."""
from __future__ import annotations

import numpy as np
from rasterio.transform import from_origin


def hip_roof(L=16.0, W=10.0, pitch=35.0, eave=6.0, rot=30.0, res=0.1, size=30.0, chimney=True, seed=0):
    """Return dsm, transform, truth dict.  Hip roof: 4 faces all at `pitch`, rotated by `rot` deg."""
    n = int(size / res)
    tr = from_origin(0, size, res, res)
    rr, cc = np.mgrid[0:n, 0:n]
    x = (cc + 0.5) * res - size / 2
    y = size / 2 - (rr + 0.5) * res
    th = np.radians(rot)
    u = x * np.cos(th) + y * np.sin(th)       # along ridge
    v = -x * np.sin(th) + y * np.cos(th)      # across ridge
    t = np.tan(np.radians(pitch))
    inside = (np.abs(u) <= L / 2) & (np.abs(v) <= W / 2)
    h = eave + t * np.minimum(W / 2 - np.abs(v), L / 2 - np.abs(u))
    z = np.where(inside, h, 0.0)
    truth = {"pitch": pitch, "aspects": sorted([(90 - rot + k * 90) % 360 for k in range(4)]), "n_faces": 4}
    if chimney:
        ch = (np.abs(u - 3.0) < 0.4) & (np.abs(v - 2.0) < 0.4)
        z = np.where(ch, z + 1.2, z)
        truth["chimney_center_uv"] = (3.0, 2.0)
    rng = np.random.default_rng(seed)
    z = z + rng.normal(0, 0.01, z.shape)
    return z, tr, truth, inside
