"""Rooftop obstruction detection from two complementary cues.

1. **Geometric (DSM) cue** - anything that sticks out of the fitted roof
   plane: residual ``r = z_DSM - z_plane > min_protrusion``.  This finds
   chimneys, vents, raised lanterns and dormers.  Faces that are themselves
   not planar (high RMSE) are dormer cheeks/roofs or other small structures
   and are flagged as such.

2. **Radiometric (RGB) cue** - flush roof windows (e.g. Velux) have no
   height signature at 10 cm DSM resolution, so they are found as colour
   anomalies *relative to their own face*: CIE-Lab distance from the face's
   robust median colour, with a robust (MAD) threshold.  Glass/metal is
   separated from cast shadow by its chromatic shift: glazing reflects the
   sky and is *less red* (Δa* < 0) and/or brighter, while a shadow mostly
   lowers lightness L* and keeps the tile chroma.

Each detection becomes a polygon (map coordinates) with a class, height above
the roof and source cue.  Shadows are reported but are not obstructions.
"""
from __future__ import annotations

import numpy as np
from rasterio import features
from scipy import ndimage as ndi
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.color import rgb2lab

from .building import excess_green


def _components_to_polys(mask, transform, res, min_area):
    lab, n = ndi.label(mask)
    out = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() * res * res < min_area:
            continue
        geoms = [shape(g) for g, v in features.shapes(m.astype(np.uint8), mask=m, transform=transform)]
        g = unary_union(geoms)
        out.append((m, g))
    return out


def detect_obstructions(seg, labels, planes, faces, rgb_full, transform_crop, cfg, veg_threshold=0.05):
    res = seg.res
    z_roof = seg.roof_surface(labels, planes)
    resid = np.where(seg.mask, seg.z - z_roof, 0.0)
    face_by_id = {f.id: f for f in faces}
    obstacles = []

    # ---------------------------------------------------------------- DSM protrusions
    prot = seg.mask & (resid > cfg.min_protrusion)
    prot = ndi.binary_opening(prot, iterations=1)
    for m, g in _components_to_polys(prot, transform_crop, res, cfg.min_area):
        h = float(np.percentile(resid[m], 95))
        area = g.area
        kind = "chimney/vent" if area < 2.5 else "dormer/raised structure"
        obstacles.append({"geom": g.buffer(0.05), "kind": kind, "height": h, "area": area, "source": "dsm"})

    # ---------------------------------------------------------------- non-planar faces
    for f in faces:
        a = f.area_px * res * res
        if f.extra["p95_abs_residual"] > 0.35 or (a < cfg.small_face_area and f.slope > 60):
            m = labels == f.id
            g = unary_union([shape(gg) for gg, v in features.shapes(m.astype(np.uint8), mask=m, transform=transform_crop)])
            obstacles.append({"geom": g, "kind": "dormer/raised structure", "height": float(np.percentile(resid[m], 95)),
                              "area": g.area, "source": "dsm-nonplanar-face", "face": f.id})
            f.extra["structure"] = True

    # ---------------------------------------------------------------- RGB colour anomalies
    r0, r1, c0, c1 = seg.window
    rgb = rgb_full[r0:r1, c0:c1]
    lab_img = rgb2lab(rgb)
    roof_med = np.median(lab_img[seg.mask], 0)
    d_glob = lab_img - roof_med
    # (a) roof-wide glazing cue: clearly less red than the roofing material
    glass_glob = seg.mask & (d_glob[..., 1] < cfg.glass_delta_a)
    edge_px = max(1, int(round(cfg.edge_margin / res)))
    glass = glass_glob.copy()
    shadow = np.zeros(labels.shape, bool)
    for f in faces:
        m = labels == f.id
        if f.extra.get("structure"):
            continue
        # (b) whole face is glazing (e.g. facets of a roof lantern)
        frac = glass_glob[m].mean() if m.any() else 0
        f.extra["glass_fraction"] = float(frac)
        if frac > cfg.glazed_face_fraction:
            f.extra["glazed"] = True
            glass |= m
            continue
        # (c) per-face anomaly: robust z-score of the Lab difference to the face's own median
        core = ndi.binary_erosion(m, iterations=edge_px)
        if core.sum() < 50:
            continue
        med = np.median(lab_img[core], 0)
        d = lab_img - med
        sig = 1.4826 * np.median(np.abs(d[core]), 0) + 0.5
        zsc = np.linalg.norm(d / sig, axis=-1)
        de = np.linalg.norm(d, axis=-1)
        anom = core & (zsc > cfg.rgb_k_mad) & (de > cfg.rgb_min_delta_e)
        is_shadow = (d[..., 0] < -12.0) & (np.abs(d[..., 1]) <= 3.0)
        glass |= anom & ~is_shadow & ((d[..., 1] < -2.5) | (d[..., 0] > 12.0) | (d[..., 2] < -6.0))
        shadow |= anom & is_shadow
        # (d) chromatic anomaly in the a*b* plane only (illumination-invariant-ish):
        # glazing reflects sky -> shifts towards blue/cyan without getting darker;
        # a shadow shifts lightness strongly, so it is gated out by ΔL.
        dc = np.hypot(d[..., 1], d[..., 2])
        sc = 1.4826 * np.median(np.abs(dc[core] - np.median(dc[core]))) + 0.5
        chrom = core & ((dc - np.median(dc[core])) / sc > cfg.chroma_k_mad) & (dc > cfg.chroma_min) \
            & (d[..., 0] > -8.0) & ((d[..., 1] < -2.0) | (d[..., 2] < -5.0))
        glass |= chrom
    glass = ndi.binary_closing(ndi.binary_opening(glass, iterations=1), iterations=2) & seg.mask
    for m, g in _components_to_polys(glass, transform_crop, res, cfg.rgb_min_area):
        hull = g.convex_hull
        solidity = g.area / max(hull.area, 1e-9)
        rect = g.minimum_rotated_rectangle
        rectangularity = g.area / max(rect.area, 1e-9)
        glazed_face = any(f.extra.get("glazed") and (labels[m] == f.id).mean() > 0.2 for f in faces)
        if not glazed_face and (g.area > cfg.rgb_max_area or solidity < 0.6):
            continue
        h = float(np.percentile(resid[m], 95))
        raised = h > cfg.min_protrusion or glazed_face
        kind = "skylight/lantern (raised glazing)" if raised else "roof window (flush)"
        geom = rect if rectangularity > 0.6 else hull
        obstacles.append({"geom": geom.buffer(0.05), "kind": kind, "height": h, "area": g.area, "source": "rgb",
                          "rectangularity": rectangularity})

    # vegetation overhanging the roof (DSM protrusion that is green in RGB): shading obstacle
    exg = excess_green(rgb)
    for o in obstacles:
        if o["source"] == "dsm" or (o["height"] > 1.5 and "lantern" not in o["kind"]) or o["height"] > 3.0:
            from rasterio.features import rasterize
            mm = rasterize([o["geom"].buffer(0.5)], out_shape=labels.shape, transform=transform_crop).astype(bool)
            if mm.any() and (exg[mm] > veg_threshold).mean() > 0.25:
                o["kind"] = "overhanging vegetation"

    # merge DSM + RGB detections that describe the same object (glass wins the label)
    obstacles = _merge(obstacles)
    shadows = [g for _, g in _components_to_polys(ndi.binary_opening(shadow, iterations=2), transform_crop, res, 0.3)]
    return obstacles, resid, glass, shadows


def _merge(obs):
    """Union detections of the same physical object; decide its class from the
    constituents: glazing only if glass covers most of the merged footprint."""
    GLASS = ("skylight/lantern (raised glazing)", "roof window (flush)")
    groups = []
    for o in obs:
        hits = [g for g in groups if any(q["geom"].intersection(o["geom"]).area > 0.3 * min(q["geom"].area, o["geom"].area)
                                         for q in g)]
        if not hits:
            groups.append([o])
        else:
            hits[0].append(o)
            for h in hits[1:]:
                hits[0].extend(h)
                groups.remove(h)
    out = []
    for g in groups:
        geom = unary_union([q["geom"] for q in g])
        glass = [q for q in g if q["kind"] in GLASS]
        other = [q for q in g if q["kind"] not in GLASS]
        glass_area = unary_union([q["geom"] for q in glass]).area if glass else 0.0
        height = max(q["height"] for q in g)
        if glass and (glass_area > 0.5 * geom.area or not other):
            kind = "skylight/lantern (raised glazing)" if (height > 0.2 or any(q["kind"] == GLASS[0] for q in glass)) else GLASS[1]
        elif any(q["kind"] == "overhanging vegetation" for q in other):
            kind = "overhanging vegetation"
        else:
            kind = "dormer/raised structure" if geom.area >= 2.5 or any(q["kind"].startswith("dormer") for q in other) \
                else "chimney/vent"
        out.append({"geom": geom, "kind": kind, "height": height, "area": geom.area,
                    "source": "+".join(sorted({q["source"] for q in g}))})
    for i, o in enumerate(out):
        o["id"] = i + 1
    return out
