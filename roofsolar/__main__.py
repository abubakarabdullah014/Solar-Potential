"""Command line entry point:  python -m roofsolar --dsm ... --rgb ... --clip ... --out outputs"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import numpy as np
from rasterio.features import rasterize
from shapely.ops import unary_union

from . import viz
from .config import Config
from .mesh3d import build_model, plotly_html
from .pipeline import run, save_outputs, roof_surface_full


def main(argv=None):
    ap = argparse.ArgumentParser(description="3-D roof reconstruction and PV module layout from DSM + RGB")
    ap.add_argument("--dsm", default="data/dsm_full.tif", help="full DSM GeoTIFF (reference grid)")
    ap.add_argument("--rgb", default="data/rgb.tif", help="RGB orthophoto GeoTIFF")
    ap.add_argument("--clip", default="data/dsm_clipped.tif", help="clipped DSM used as parcel / AOI mask")
    ap.add_argument("--out", default="outputs", help="output directory")
    ap.add_argument("--config", help="JSON file overriding config values, e.g. {\"solar\": {\"edge_setback\": 0.4}}")
    ap.add_argument("--no-render", action="store_true", help="skip headless-browser 3-D PNG renders")
    ap.add_argument("--no-shading", action="store_true", help="skip DSM ray-cast shading analysis")
    ap.add_argument("--turntable", action="store_true", help="also write a rotating 3-D MP4 (needs ffmpeg)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")

    cfg = Config()
    if a.config:
        for section, vals in json.loads(Path(a.config).read_text()).items():
            for k, v in vals.items():
                setattr(getattr(cfg, section), k, v)

    t0 = time.time()
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    R = run(a.dsm, a.rgb, a.clip, cfg, shading=not a.no_shading)
    report = save_outputs(R, out)

    roof, _ = roof_surface_full(R)
    viz.fig_inputs(R, out / "figures/01_inputs.png")
    viz.fig_slope_aspect(R, out / "figures/02_elevation_slope_aspect.png")
    viz.fig_segmentation(R, out / "figures/03_roof_faces_and_edges.png")
    viz.fig_regularization(R, out / "figures/04_regularization.png", roof)
    viz.fig_obstructions(R, out / "figures/05_obstructions.png")
    viz.fig_panels(R, out / "figures/06_panel_layout.png")

    sc = R.scene

    def roof_z_at(poly):
        m = rasterize([poly], out_shape=sc.shape, transform=sc.transform).astype(bool)
        v = roof[m]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (R.binfo["ground_height"],) * 2
        return float(v.min()), float(np.percentile(R.dsm_clean[m], 95))

    c = unary_union([f.polygon for f in R.faces]).centroid
    origin = (c.x, c.y)
    mb = build_model(R.faces, R.obstacles, R.panels, R.binfo["ground_height"], roof_z_at, origin)
    (out / "model").mkdir(exist_ok=True)
    mb.export(out / "model/roof_model.glb")
    mb.scene().export(out / "model/roof_model.obj")
    plotly_html(mb, out / "model/roof_model_3d.html",
                title=f"Reconstructed roof + {len(R.panels)} PV modules (local frame, origin E{origin[0]:.1f} N{origin[1]:.1f})")
    (out / "model/origin.json").write_text(json.dumps({"crs": str(sc.crs), "origin_xy": origin}))
    if not a.no_render:
        try:
            from .mesh3d import render_png
            render_png(mb, out / "figures/07_3d_perspective.png", camera=dict(eye=dict(x=-0.75, y=-1.05, z=0.45)))
            render_png(mb, out / "figures/08_3d_top.png", camera=dict(eye=dict(x=0, y=0, z=2.3), up=dict(x=0, y=1, z=0)))
            if a.turntable:
                from .mesh3d import render_turntable
                render_turntable(mb, out / "model/turntable.mp4")
        except Exception as e:  # headless browser missing - not fatal
            logging.warning("3-D PNG render skipped: %s", e)
            viz.fig_3d(mb, out / "figures/07_3d_perspective.png")
    with open(out / "result.pkl", "wb") as fh:
        pickle.dump(R, fh)

    rec = report["metrics"]["reconstruction"]
    pv = report["metrics"]["panels"]
    print(f"faces: {rec['n_faces']}  roof RMSE {rec['roof_rmse_m']:.3f} m  (MAE {rec['roof_mae_m']:.3f} m, "
          f"{rec['within_10cm_pct']:.1f}% within 10 cm)  seam gap p95 {rec['seam_gap_p95_m']:.2e} m")
    print(f"obstructions: {report['n_obstructions']}  {report['obstructions_by_kind']}")
    print(f"PV: {report['n_panels']} modules = {report['system_kWp']:.2f} kWp   mean beam shading loss "
          f"{100 * report['mean_shading_loss']:.1f}%   constraint check: {'PASS' if pv['passed'] else 'FAIL'} "
          f"({pv['n_violations']} violations)")
    print(f"outputs in {out.resolve()}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
