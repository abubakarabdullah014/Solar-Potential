import math

import numpy as np
import pytest
from shapely.geometry import box, Polygon

from roofsolar.config import Config
from roofsolar.geometry import PlaneFrame, plane_intersection_line
from roofsolar.preprocess import harmonic_inpaint, detect_outliers
from roofsolar.segmentation import Plane, RoofSegmenter, fit_plane_robust, build_faces
from roofsolar.solar import pack_region, sun_positions
from roofsolar.synthetic import hip_roof
from roofsolar.terrain import slope_aspect


def _plane_grid(a, b, c, n=50, res=0.1):
    rr, cc = np.mgrid[0:n, 0:n]
    x, y = cc * res, -rr * res
    return a * x + b * y + c


@pytest.mark.parametrize("a,b,exp_aspect", [(0, 0.5, 180.0), (0, -0.5, 0.0), (0.5, 0, 270.0), (-0.5, 0, 90.0)])
def test_horn_slope_aspect_on_plane(a, b, exp_aspect):
    z = _plane_grid(a, b, 100)
    s, asp, *_ = slope_aspect(z, 0.1)
    core = (slice(5, -5), slice(5, -5))
    assert np.allclose(s[core], math.degrees(math.atan(math.hypot(a, b))), atol=1e-6)
    assert np.allclose(asp[core], exp_aspect, atol=1e-6)


def test_harmonic_inpaint_is_exact_for_planes():
    z = _plane_grid(0.3, -0.7, 50)
    miss = np.zeros_like(z, bool)
    miss[10:30, 15:35] = True
    miss[np.random.default_rng(0).random(z.shape) < 0.2] = True
    miss[0, :] = miss[-1, :] = miss[:, 0] = miss[:, -1] = False
    out = harmonic_inpaint(np.where(miss, 0, z), miss)
    assert np.abs(out - z).max() < 1e-8


def test_outlier_detection_finds_spikes_only():
    z = _plane_grid(0.2, 0.1, 50, n=80)
    z[40, 40] += 3.0
    z[20, 60] -= 2.0
    m = detect_outliers(z)
    assert m[40, 40] and m[20, 60] and m.sum() == 2


def test_robust_plane_fit_ignores_outliers():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 10, 2000), rng.uniform(0, 10, 2000)
    z = 0.7 * x - 0.2 * y + 5 + rng.normal(0, 0.02, x.size)
    z[:400] += rng.uniform(0.5, 2.0, 400)  # 20 % chimney-like outliers
    p, _ = fit_plane_robust(x, y, z)
    assert abs(p.a - 0.7) < 0.01 and abs(p.b + 0.2) < 0.01 and abs(p.c - 5) < 0.05


def test_plane_intersection_line():
    p1, p2 = Plane(0.7, 0, 0), Plane(-0.7, 0, 0)        # gable with ridge on x = 0
    p0, d = plane_intersection_line(p1, p2)
    assert abs(p0[0]) < 1e-9 and abs(abs(d[1]) - 1) < 1e-9


@pytest.mark.parametrize("a,b", [(0.7, 0.0), (0.3, -0.5), (0.0, 0.0)])
def test_plane_frame_preserves_true_lengths(a, b):
    pl = Plane(a, b, 10)
    fr = PlaneFrame(pl, (100.0, 200.0), eave_azimuth_deg=0 if a == b == 0 else None)
    uv = np.array([[0, 0], [1.722, 0], [1.722, 1.134], [0, 1.134]])
    xyz = fr.to_xyz(uv)
    e = [np.linalg.norm(xyz[(i + 1) % 4] - xyz[i]) for i in range(4)]
    assert np.allclose(e, [1.722, 1.134, 1.722, 1.134], atol=1e-9)
    # right angles in 3-D
    assert abs(np.dot(xyz[1] - xyz[0], xyz[3] - xyz[0])) < 1e-9
    assert np.allclose(fr.to_uv(fr.to_xy(uv)), uv)
    if a or b:  # u axis is horizontal (along eave)
        assert abs(xyz[1, 2] - xyz[0, 2]) < 1e-9


def test_packing_exact_on_rectangle():
    lays = pack_region(box(0, 0, 10.0, 4.0), 1.0, 2.0, 0.0, 0.05)
    assert max(l.n for l in lays) == 20
    # with a 2 cm gap only 9 columns (9*1.02-0.02 = 9.16 m) and 1 row (2*2.02-0.02 = 4.02 > 4) fit
    lays = pack_region(box(0, 0, 10.0, 4.0), 1.0, 2.0, 0.02, 0.01)
    assert max(l.n for l in lays) == 9


def test_packing_respects_holes():
    region = box(0, 0, 6, 2).difference(box(2.5, 0.5, 3.5, 1.5))
    lays = pack_region(region, 1.0, 2.0, 0.0, 0.05)
    for lay in lays:
        for u, v in lay.panels_uv:
            assert region.buffer(1e-9).contains(box(u, v, u + 1, v + 2))
    assert max(l.n for l in lays) == 4      # 2 left + 2 right of the hole


def test_sun_position_sanity():
    zen, az, D = sun_positions(52.0, 0.0)
    noon = np.argmin(zen[D == 80])          # ~ March equinox
    assert abs(np.degrees(zen[D == 80][noon]) - 52.0) < 1.5
    assert abs(np.degrees(az[D == 80][noon]) - 180.0) < 8


def test_synthetic_hip_roof_reconstruction():
    z, tr, truth, inside = hip_roof()
    cfg = Config().segmentation
    seg = RoofSegmenter(z, inside, 0.1, cfg)
    labels, planes, edges = seg.run()
    faces = build_faces(seg, labels, planes, tr, cfg.flat_slope)
    big = [f for f in faces if f.area_px * 0.01 > 5]
    assert len(big) == 4
    for f in big:
        assert abs(f.slope - truth["pitch"]) < 0.5
        assert min(abs((f.aspect - t + 180) % 360 - 180) for t in truth["aspects"]) < 0.5
    kinds = {k for k in edges.values()}
    assert "convex" in kinds
