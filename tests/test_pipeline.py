"""Integration test on the assessment data (skipped if the rasters are not present)."""
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
pytestmark = pytest.mark.skipif(not (DATA / "dsm_full.tif").exists(), reason="assessment data not available")


@pytest.fixture(scope="module")
def result():
    from roofsolar.pipeline import run
    return run(DATA / "dsm_full.tif", DATA / "rgb.tif", DATA / "dsm_clipped.tif", shading=True)


def test_reconstruction_quality(result):
    m = result.metrics["reconstruction"]
    assert m["n_faces"] >= 10
    assert m["roof_mae_m"] < 0.10
    assert m["within_20cm_pct"] > 90
    assert m["seam_gap_p95_m"] < 1e-3          # watertight ridges / hips / valleys


def test_panel_constraints_hold(result):
    v = result.metrics["panels"]
    assert v["n_panels"] > 0
    assert v["passed"], v["violations"][:5]


def test_obstructions_found(result):
    kinds = {o["kind"] for o in result.obstacles}
    assert "chimney/vent" in kinds
    assert any("glazing" in k or "window" in k for k in kinds)


def test_no_panels_on_poorly_oriented_faces(result):
    summ = {s["face"]: s for s in result.face_summary}
    for p in result.panels:
        assert summ[p["face"]]["relative_yield"] >= result.cfg.solar.min_relative_yield
