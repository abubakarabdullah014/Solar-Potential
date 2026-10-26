"""End-to-end pipeline: DSM + RGB  ->  roof model + obstructions + PV layout."""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rasterio import Affine
from rasterio.features import rasterize
from shapely.geometry import mapping

from .building import extract_building
from .config import Config
from .geometry import label_polygons, simplify_coverage, shared_edge_lines, snap_to_plane_intersections
from .io import load_scene, write_geotiff, Scene
from .obstructions import detect_obstructions
from .preprocess import preprocess_dsm
from .segmentation import RoofSegmenter, build_faces
from .solar import place_panels, shading_loss
from .terrain import slope_aspect
from .validation import reconstruction_metrics, validate_panels

log = logging.getLogger("roofsolar")


@dataclass
class Result:
    scene: Scene
    cfg: Config
    dsm_clean: np.ndarray = None
    pre: dict = None
    slope: np.ndarray = None
    aspect: np.ndarray = None
    building: np.ndarray = None
    dtm: np.ndarray = None
    ndsm: np.ndarray = None
    veg: np.ndarray = None
    binfo: dict = None
    seg: RoofSegmenter = None
    labels: np.ndarray = None
    planes: dict = None
    edges: dict = None
    faces: list = None
    edge_lines: list = None
    obstacles: list = None
    residual: np.ndarray = None
    glass: np.ndarray = None
    shadows: list = None
    panels: list = None
    face_summary: list = None
    metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    dominant_azimuth: float = 0.0
    lat: float = 0.0
    rejected_shaded: list = field(default_factory=list)
    lon: float = 0.0

    @property
    def crop_transform(self):
        r0, _, c0, _ = self.seg.window
        return self.scene.transform * Affine.translation(c0, r0)


def _seam_gaps(polys, planes, edges):
    import shapely
    g = []
    for (u, v), kind in edges.items():
        if kind == "step" or u not in polys or v not in polys:
            continue
        pts = shapely.get_coordinates(polys[u].boundary.intersection(polys[v].boundary))
        if len(pts):
            g.extend(np.abs(planes[u].z(pts[:, 0], pts[:, 1]) - planes[v].z(pts[:, 0], pts[:, 1])))
    g = np.array(g) if g else np.zeros(1)
    return {"median_m": float(np.median(g)), "p95_m": float(np.percentile(g, 95)), "max_m": float(g.max())}


def dominant_orientation(faces):
    """Building's principal axis from area-weighted face aspects (mod 90°, circular mean of 4θ)."""
    th = np.radians([f.aspect for f in faces if not f.is_flat])
    w = np.array([f.area_px for f in faces if not f.is_flat], float)
    if len(th) == 0:
        return 0.0
    ang = math.atan2((w * np.sin(4 * th)).sum(), (w * np.cos(4 * th)).sum()) / 4
    return math.degrees(ang) % 90


def run(dsm_full, rgb, dsm_clipped=None, cfg: Config | None = None, shading=True) -> Result:
    cfg = cfg or Config()
    T = {}
    t = time.time()
    scene = load_scene(dsm_full, rgb, dsm_clipped)
    R = Result(scene=scene, cfg=cfg)
    lon, lat = scene.lonlat_center()
    R.lat = cfg.solar.latitude or lat
    R.lon = cfg.solar.longitude or lon
    T["load"] = time.time() - t

    t = time.time()
    R.dsm_clean, R.pre = preprocess_dsm(scene.dsm, cfg.preprocess)
    R.slope, R.aspect, _, _ = slope_aspect(R.dsm_clean, scene.res)
    T["preprocess"] = time.time() - t
    log.info("preprocess: %s", R.pre["report"])

    t = time.time()
    R.building, R.dtm, R.ndsm, R.veg, R.binfo = extract_building(R.dsm_clean, scene.rgb, scene.parcel, scene.res, cfg.building)
    T["building"] = time.time() - t
    log.info("building: %s", R.binfo)

    t = time.time()
    seg = RoofSegmenter(R.dsm_clean, R.building, scene.res, cfg.segmentation)
    labels, planes, edges = seg.run()
    faces = build_faces(seg, labels, planes, scene.transform, cfg.segmentation.flat_slope)
    R.seg, R.labels, R.planes, R.edges = seg, labels, planes, edges
    full_labels = seg.uncrop(labels)
    raw = label_polygons(full_labels, scene.transform, set(planes))
    simp = simplify_coverage(raw, cfg.segmentation.simplify_tolerance)
    world_planes = {f.id: f.plane for f in faces}
    R.metrics["seam_gap_before_snap"] = _seam_gaps(simp, world_planes, edges)
    # an adjacency whose analytic intersection line is far from the observed shared
    # boundary is not a real ridge/hip/valley -> treat it as a vertical step edge
    for e in shared_edge_lines(simp, world_planes, edges):
        if e.get("fit_dist", 0) > cfg.segmentation.max_seam_offset:
            edges[tuple(sorted(map(int, e["faces"])))] = "step"
    simp, R.metrics["snap"] = snap_to_plane_intersections(simp, world_planes, edges, cfg.segmentation.max_seam_offset)
    faces = [f for f in faces if f.id in simp]
    for f in faces:
        f.polygon = simp[f.id]
    R.faces = faces
    R.edge_lines = shared_edge_lines(simp, {f.id: f.plane for f in faces}, edges)
    R.dominant_azimuth = dominant_orientation(faces)
    T["segmentation"] = time.time() - t
    log.info("segmentation: %d faces", len(faces))

    t = time.time()
    R.obstacles, R.residual, R.glass, R.shadows = detect_obstructions(
        seg, labels, planes, faces, scene.rgb, R.crop_transform, cfg.obstruction, R.binfo["exg_threshold"])
    T["obstructions"] = time.time() - t
    log.info("obstructions: %d", len(R.obstacles))

    t = time.time()
    R.panels, R.face_summary = place_panels(faces, R.obstacles, cfg.solar, R.dominant_azimuth + 90, R.lat, R.lon)
    T["placement"] = time.time() - t
    if shading and R.panels:
        t = time.time()
        loss = shading_loss(R.panels, R.dsm_clean, scene.transform, R.lat, R.lon)
        for p, l in zip(R.panels, loss):
            p["shading_loss"] = float(l)
        R.rejected_shaded = [p for p in R.panels if p["shading_loss"] > cfg.solar.max_shading_loss]
        R.panels = [p for p in R.panels if p["shading_loss"] <= cfg.solar.max_shading_loss]
        for s in R.face_summary:
            n = sum(p["face"] == s["face"] for p in R.panels)
            if s.get("status") == "used" and n != s["n_panels"]:
                s["n_panels_before_shading"] = s["n_panels"]
                s["n_panels"] = n
                s["kWp"] = n * cfg.solar.panel_power_wp / 1000
                if n == 0:
                    s["status"] = "excluded: shaded (> %.0f%% beam loss)" % (100 * cfg.solar.max_shading_loss)
        T["shading"] = time.time() - t
    log.info("placement: %d panels", len(R.panels))

    t = time.time()
    R.metrics["reconstruction"] = reconstruction_metrics(seg, labels, planes, faces, R.edge_lines)
    R.metrics["panels"] = validate_panels(R.panels, faces, R.obstacles, cfg.solar)
    T["validation"] = time.time() - t
    R.timings = T
    return R


# --------------------------------------------------------------------------- outputs
def roof_surface_full(R: Result):
    """Regularised DSM on the full grid: roof replaced by its planes, obstructions kept from data."""
    zr_crop = R.seg.roof_surface(R.labels, R.planes)
    roof = R.seg.uncrop(np.nan_to_num(zr_crop, nan=-9999.0), fill=-9999.0)
    roof[roof == -9999.0] = np.nan
    reg = R.dsm_clean.copy()
    inb = np.isfinite(roof)
    reg[inb] = roof[inb]
    obs_mask = np.zeros(R.scene.shape, bool)
    if R.obstacles:
        obs_mask = rasterize([o["geom"] for o in R.obstacles if o["kind"] != "roof window (flush)"],
                             out_shape=R.scene.shape, transform=R.scene.transform).astype(bool)
    keep = obs_mask & inb & (R.dsm_clean > roof)
    reg[keep] = R.dsm_clean[keep]
    return roof, reg


def _feature(geom, props):
    return {"type": "Feature", "geometry": mapping(geom), "properties": props}


def _fc(features, crs):
    return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": str(crs)}}, "features": features}


def save_outputs(R: Result, out: Path):
    out = Path(out)
    (out / "rasters").mkdir(parents=True, exist_ok=True)
    (out / "vectors").mkdir(exist_ok=True)
    (out / "model").mkdir(exist_ok=True)
    sc = R.scene
    roof, reg = roof_surface_full(R)
    write_geotiff(out / "rasters/dsm_clean.tif", R.dsm_clean, sc, dtype="float32")
    write_geotiff(out / "rasters/slope_deg.tif", R.slope, sc, dtype="float32")
    write_geotiff(out / "rasters/aspect_deg.tif", np.nan_to_num(R.aspect, nan=-1), sc, nodata=-1, dtype="float32")
    write_geotiff(out / "rasters/ndsm.tif", R.ndsm, sc, dtype="float32")
    write_geotiff(out / "rasters/roof_faces.tif", R.seg.uncrop(R.labels), sc, nodata=0, dtype="int32")
    write_geotiff(out / "rasters/roof_planes_dsm.tif", np.nan_to_num(roof, nan=-9999), sc, nodata=-9999, dtype="float32")
    write_geotiff(out / "rasters/dsm_regularized.tif", reg, sc, dtype="float32")

    summ = {s["face"]: s for s in R.face_summary}
    feats = []
    for f in R.faces:
        s = summ.get(f.id, {})
        feats.append(_feature(f.polygon, {"face": f.id, "slope_deg": round(f.slope, 2), "aspect_deg": round(f.aspect, 2),
                                          "plane_a": f.plane.a, "plane_b": f.plane.b, "plane_c": f.plane.c,
                                          "rmse_m": round(f.rmse, 4), "status": s.get("status", ""),
                                          "n_panels": s.get("n_panels", 0),
                                          "relative_yield": round(s.get("relative_yield", 0), 3)}))
    (out / "vectors/roof_faces.geojson").write_text(json.dumps(_fc(feats, sc.crs)))
    feats = [_feature(e["geom2d"], {"faces": list(map(int, e["faces"])), "kind": e["kind"]}) for e in R.edge_lines]
    (out / "vectors/roof_edges.geojson").write_text(json.dumps(_fc(feats, sc.crs)))
    feats = [_feature(o["geom"], {"id": o["id"], "kind": o["kind"], "height_m": round(o["height"], 2),
                                  "area_m2": round(o["area"], 2), "source": o["source"]}) for o in R.obstacles]
    (out / "vectors/obstructions.geojson").write_text(json.dumps(_fc(feats, sc.crs)))
    feats = []
    for k, p in enumerate(R.panels):
        feats.append(_feature(p["footprint"], {"panel": k, "face": p["face"], "orientation": p["orientation"],
                                               "corners_xyz": np.round(p["xyz"], 3).tolist(),
                                               "shading_loss": round(p.get("shading_loss", 0), 3)}))
    (out / "vectors/panels.geojson").write_text(json.dumps(_fc(feats, sc.crs)))

    # per-face CSV
    import csv
    keys = ["face", "status", "slope", "aspect", "area_plan_m2", "area_true_m2", "usable_true_m2", "relative_yield",
            "n_panels", "orientation", "mode", "kWp"]
    with open(out / "face_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for s in sorted(R.face_summary, key=lambda s: s["face"]):
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})

    sl = [p.get("shading_loss", 0) for p in R.panels]
    report = {
        "location": {"lat": R.lat, "lon": R.lon, "crs": str(sc.crs), "resolution_m": sc.res},
        "preprocess": R.pre["report"],
        "building": R.binfo,
        "dominant_axis_deg": R.dominant_azimuth,
        "n_obstructions": len(R.obstacles),
        "obstructions_by_kind": _count([o["kind"] for o in R.obstacles]),
        "n_panels": len(R.panels),
        "system_kWp": len(R.panels) * R.cfg.solar.panel_power_wp / 1000,
        "mean_shading_loss": float(np.mean(sl)) if sl else 0.0,
        "metrics": R.metrics,
        "timings_s": R.timings,
        "config": R.cfg.to_dict(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def _count(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d
