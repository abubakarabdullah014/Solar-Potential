"""Quantitative self-checks.  Every number here is computed, none is asserted by hand.

Reconstruction
  * per-face RMSE / MAE / p95 of DSM - plane
  * share of roof pixels explained within 10 cm / 20 cm
  * seam gap: |z_A - z_B| at shared vertices of adjacent faces with a
    ridge/hip/valley edge (0 for a perfectly watertight reconstruction)
  * ridge/hip line fit: distance between raster boundary and the analytic
    plane-intersection line

Panels (hard constraints - any violation is a bug)
  * true module edge lengths equal the spec (validates the slope correction)
  * all 4 corners lie on the face plane
  * rectangle inside the usable region (setback & keep-outs included)
  * no module overlaps any other module
  * distance to every obstruction >= clearance (measured in the roof plane)
  * distance to face edge >= setback
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.strtree import STRtree


def reconstruction_metrics(seg, labels, planes, faces, edges_lines):
    m = seg.mask & (labels > 0)
    zr = seg.roof_surface(labels, planes)
    r = (seg.z - zr)[m]
    face_poly = {f.id: f.polygon for f in faces}
    face_plane = {f.id: f.plane for f in faces}
    gaps = []
    for e in edges_lines:
        if e["kind"] == "step":
            continue
        u, v = e["faces"]
        if u not in face_poly or v not in face_poly:
            continue
        shared = face_poly[u].boundary.intersection(face_poly[v].boundary)
        pts = shapely.get_coordinates(shared)
        if len(pts):
            gaps.extend(np.abs(face_plane[u].z(pts[:, 0], pts[:, 1]) - face_plane[v].z(pts[:, 0], pts[:, 1])))
    gaps = np.array(gaps) if gaps else np.zeros(1)
    return {
        "roof_rmse_m": float(np.sqrt(np.mean(r ** 2))),
        "roof_mae_m": float(np.mean(np.abs(r))),
        "within_10cm_pct": float(100 * np.mean(np.abs(r) < 0.10)),
        "within_20cm_pct": float(100 * np.mean(np.abs(r) < 0.20)),
        "seam_gap_median_m": float(np.median(gaps)),
        "seam_gap_p95_m": float(np.percentile(gaps, 95)),
        "ridge_line_fit_max_m": float(max([e.get("fit_dist", 0) for e in edges_lines] or [0])),
        "n_faces": len(faces),
        "faces": [{"id": f.id, "area_plan_m2": round(f.polygon.area, 2), "slope": round(f.slope, 1),
                   "aspect": round(f.aspect, 1), "rmse_m": round(f.rmse, 3),
                   "p95_abs_residual_m": round(f.extra["p95_abs_residual"], 3)} for f in faces],
    }


def validate_panels(panels, faces, obstacles, cfg, tol=1e-6):
    fmap = {f.id: f for f in faces}
    errs = []
    L, W = cfg.panel_length, cfg.panel_width
    for k, p in enumerate(panels):
        xyz = p["xyz"]
        e = [np.linalg.norm(xyz[(i + 1) % 4] - xyz[i]) for i in range(4)]
        if not (np.allclose(sorted(e), sorted([W, W, L, L]), atol=1e-3)):
            errs.append(f"panel {k}: edge lengths {np.round(e, 4)}")
        f = fmap[p["face"]]
        dz = np.abs(xyz[:, 2] - f.plane.z(xyz[:, 0], xyz[:, 1]))
        if dz.max() > 1e-6:
            errs.append(f"panel {k}: off-plane by {dz.max():.2e} m")
        rect = Polygon(p["uv"])
        if not p["usable_uv"].buffer(1e-6).contains(rect):
            errs.append(f"panel {k}: outside usable region")
        frame = f.extra["frame"]
        face_uv = frame.poly_to_uv(f.polygon)
        d_edge = face_uv.exterior.distance(rect)
        for ring in face_uv.interiors:
            d_edge = min(d_edge, ring.distance(rect))
        if d_edge < cfg.edge_setback - 1e-4:
            errs.append(f"panel {k}: edge setback {d_edge:.3f} m")
        for o in obstacles:
            if o["geom"].distance(f.polygon) > 3:
                continue
            d = frame.poly_to_uv(o["geom"]).distance(rect)
            if d < cfg.obstruction_clearance - 1e-4:
                errs.append(f"panel {k}: {d:.3f} m from obstruction {o['id']}")
    fps = [p["footprint"] for p in panels]
    tree = STRtree(fps)
    n_overlap = 0
    for i, g in enumerate(fps):
        for j in tree.query(g):
            if j > i and g.intersection(fps[j]).area > tol:
                n_overlap += 1
                errs.append(f"panels {i},{j} overlap")
    return {"n_panels": len(panels), "n_violations": len(errs), "violations": errs[:50], "n_overlaps": n_overlap,
            "passed": len(errs) == 0}
