"""Robustness of the reconstruction to noise and missing data.

The real DSM is corrupted in controlled ways (Gaussian noise, random pixel
drop-out, large rectangular holes on the roof, spikes) and the whole pipeline is
re-run.  The clean-data reconstruction is the reference.  For every major face
(> 5 m^2) the matching face (max polygon IoU) is found and we report:
  * angle between plane normals, |Δslope|, |Δaspect|
  * RMSE between the two piece-wise planar roof surfaces
  * PV module count
Run:  python experiments/robustness.py  (writes outputs/robustness.csv/.md/.png)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roofsolar.pipeline import run, roof_surface_full  # noqa: E402
from roofsolar.segmentation import angle_between  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "outputs"


def corrupt(z, kind, rng, roof_mask):
    z = z.copy()
    if "noise5" in kind:
        z += rng.normal(0, 0.05, z.shape)
    if "noise10" in kind:
        z += rng.normal(0, 0.10, z.shape)
    if "drop20" in kind:
        z[rng.random(z.shape) < 0.20] = np.nan
    if "holes" in kind:
        rr, cc = np.nonzero(roof_mask)
        for _ in range(6):  # 6 holes of 2 x 2 m on the roof
            k = rng.integers(len(rr))
            z[rr[k] - 10: rr[k] + 10, cc[k] - 10: cc[k] + 10] = np.nan
    if "spikes" in kind:
        m = rng.random(z.shape) < 0.01
        z[m] += rng.choice([-3.0, 3.0], m.sum())
    return z


def main():
    ref = run(DATA / "dsm_full.tif", DATA / "rgb.tif", DATA / "dsm_clipped.tif", shading=False)
    ref_roof, _ = roof_surface_full(ref)
    with rasterio.open(DATA / "dsm_full.tif") as ds:
        prof = ds.profile
        z0 = ds.read(1).astype(float)
    big = [f for f in ref.faces if f.polygon.area > 5]
    scen = ["noise5", "noise10", "drop20", "holes", "spikes", "noise5+drop20+holes+spikes"]
    rows = []
    for s in scen:
        errs = {k: [] for k in ("ang", "slope", "aspect")}
        seeds = [0, 1, 2]
        n_pan, rmse, nf = [], [], []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            zc = corrupt(z0, s, rng, ref.building)
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "dsm.tif"
                pr = dict(prof, dtype="float32", nodata=-9999.0)
                with rasterio.open(p, "w", **pr) as ds:
                    ds.write(np.where(np.isfinite(zc), zc, -9999.0).astype("float32"), 1)
                R = run(p, DATA / "rgb.tif", DATA / "dsm_clipped.tif", shading=False)
            roof, _ = roof_surface_full(R)
            m = np.isfinite(roof) & np.isfinite(ref_roof) & ref.building
            d = np.abs(roof[m] - ref_roof[m])
            rmse.append((float(np.median(d)), float(100 * np.mean(d < 0.10))))
            n_pan.append(len(R.panels))
            nf.append(len(R.faces))
            for f in big:
                best, iou = None, 0
                for g in R.faces:
                    i = f.polygon.intersection(g.polygon).area
                    u = f.polygon.union(g.polygon).area
                    if u > 0 and i / u > iou:
                        best, iou = g, i / u
                if best is None or iou < 0.3:
                    continue
                errs["ang"].append(angle_between(f.plane.normal, best.plane.normal))
                errs["slope"].append(abs(f.slope - best.slope))
                errs["aspect"].append(abs((f.aspect - best.aspect + 180) % 360 - 180))
        rows.append({"scenario": s, "faces": f"{np.mean(nf):.1f}",
                     "normal_err_med_deg": np.median(errs["ang"]), "normal_err_p90_deg": np.percentile(errs["ang"], 90),
                     "slope_err_med_deg": np.median(errs["slope"]), "aspect_err_med_deg": np.median(errs["aspect"]),
                     "surface_mad_vs_clean_m": np.mean([r[0] for r in rmse]),
                     "surface_within10cm_pct": np.mean([r[1] for r in rmse]), "panels": f"{np.mean(n_pan):.1f} (ref {len(ref.panels)})"})
        print(rows[-1], flush=True)
    import csv
    with open(OUT / "robustness.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = ["| scenario | faces | normal err med / p90 (°) | slope err med (°) | aspect err med (°) | roof surface vs clean: median abs Δz (m) / % within 10 cm | panels (no shading filter) |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['scenario']} | {r['faces']} | {r['normal_err_med_deg']:.2f} / {r['normal_err_p90_deg']:.2f} | "
                  f"{r['slope_err_med_deg']:.2f} | {r['aspect_err_med_deg']:.2f} | {r['surface_mad_vs_clean_m']:.3f} / {r['surface_within10cm_pct']:.1f}% | {r['panels']} |")
    (OUT / "robustness.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
