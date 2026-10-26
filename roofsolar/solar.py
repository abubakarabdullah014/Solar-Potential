"""Solar resource screening and PV module placement.

Placement works in each roof face's *own* 2-D frame (see ``PlaneFrame``):
the face polygon, edge setbacks and obstruction keep-outs are mapped through
the exact affine map XY -> (u, v), where u runs along the eave and v up the
slope in true metres.  Module rectangles are therefore placed at true size
(1.722 m x 1.134 m by default) on the inclined plane and mapped back to 3-D
corner coordinates that lie exactly on the plane; their 2-D map footprint is
foreshortened by cos(slope) along the fall line, as it must be.

Packing algorithm (per face, per module orientation):
  * rows are bands of height h (+gap) stacked up the slope; the row phase
    ``φv`` is searched on a fine grid;
  * for a band, the set of u where the whole band is inside the usable region
    is computed *exactly*: the projection of each connected component of
    ``band − usable`` onto the u-axis is an interval (projection of a connected
    set), and the free set is the complement of their union;
  * columns: either a common column grid with phase ``φu`` searched as well
    ("aligned" - a tidy array), or modules packed contiguously and centred in
    each free interval ("staggered" - more modules on irregular faces).
  The layout with most modules wins; ties are broken by compactness
  (smaller convex-hull area of the array, i.e. modules kept together).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

from .geometry import PlaneFrame


# --------------------------------------------------------------------------- solar geometry
def sun_positions(lat, lon, year=2025, step_h=1.0):
    """Hourly solar zenith/azimuth for a year (NOAA simplified equations), sun above horizon only."""
    doy = np.arange(1, 366)
    hours = np.arange(0, 24, step_h) + step_h / 2
    D, H = np.meshgrid(doy, hours, indexing="ij")
    g = 2 * np.pi / 365 * (D - 1 + (H - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    tst = H * 60 + eqt + 4 * lon               # true solar time (UTC hours input)
    ha = np.radians(tst / 4 - 180)
    phi = np.radians(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    zen = np.arccos(np.clip(cosz, -1, 1))
    az = np.arctan2(np.sin(ha), np.cos(ha) * np.sin(phi) - np.tan(decl) * np.cos(phi)) + np.pi  # from north
    up = cosz > 0.02
    return zen[up], (az[up]) % (2 * np.pi), D[up]


def clear_sky(zen):
    """Meinel clear-sky beam (DNI) and a simple diffuse fraction, W/m^2."""
    am = 1.0 / np.cos(zen)
    dni = 1353.0 * 0.7 ** (am ** 0.678)
    dhi = 0.1 * dni + 20.0
    return dni, dhi


def poa_annual(tilt_deg, az_deg, sun, albedo=0.2, diffuse_fraction=0.5):
    """Relative annual plane-of-array irradiation (arbitrary units).

    Beam: clear-sky DNI x cos(incidence), integrated over every daylight hour of the
    year (geometry is exact).  Sky diffuse: isotropic, view factor (1+cos β)/2.
    Ground reflected: albedo x (1-cos β)/2.  Because cloud cover is not modelled
    hour by hour, the beam and diffuse parts are weighted by the site's *annual*
    diffuse share of global horizontal irradiation (≈0.5 for the UK) - i.e. the
    transposition factors are clear-sky geometry, the mix is climatological."""
    zen, saz, _ = sun
    dni, _ = clear_sky(zen)
    b, g = np.radians(tilt_deg), np.radians(az_deg)
    cos_i = np.cos(zen) * np.cos(b) + np.sin(zen) * np.sin(b) * np.cos(saz - g)
    beam_h = (dni * np.cos(zen)).sum()
    rb = (dni * np.clip(cos_i, 0, None)).sum() / beam_h          # beam transposition factor
    fd = diffuse_fraction
    return float((1 - fd) * rb + fd * (1 + np.cos(b)) / 2 + albedo * (1 - np.cos(b)) / 2)


def relative_yield_table(lat, lon, diffuse_fraction=0.5):
    sun = sun_positions(lat, lon)
    best = max(poa_annual(t, a, sun, diffuse_fraction=diffuse_fraction) for t in range(0, 70, 5) for a in range(90, 271, 15))
    return sun, best


# --------------------------------------------------------------------------- packing
@dataclass
class Layout:
    panels_uv: list
    orientation: str
    mode: str

    @property
    def n(self):
        return len(self.panels_uv)


def _free_intervals(usable, v0, v1, umin, umax):
    band = box(umin - 1, v0, umax + 1, v1)
    bad = band.difference(usable)
    ints = []
    for g in getattr(bad, "geoms", [bad]):
        if g.is_empty or g.area < 1e-10:
            continue
        x0, _, x1, _ = g.bounds
        ints.append((x0, x1))
    ints.sort()
    free, cur = [], umin - 1
    for a, b in ints:
        if a > cur:
            free.append((cur, a))
        cur = max(cur, b)
    if cur < umax + 1:
        free.append((cur, umax + 1))
    return [(a, b) for a, b in free if b - a > 1e-6]


def pack_region(usable: Polygon, w: float, h: float, gap: float, step: float) -> list[Layout]:
    """Return the best aligned and the best staggered layouts of w x h rectangles."""
    if usable.is_empty or usable.area < w * h:
        return []
    umin, vmin, umax, vmax = usable.bounds
    pu, pv = w + gap, h + gap
    best_al, best_st = None, None
    for phv in np.arange(0, pv, step):
        rows = []
        v = vmin + phv
        while v + h <= vmax + 1e-9:
            rows.append((v, _free_intervals(usable, v, v + h, umin, umax)))
            v += pv
        if not rows:
            continue
        # staggered: contiguous modules centred in every free interval
        st = []
        for v, ints in rows:
            for a, b in ints:
                n = int(math.floor((b - a + gap) / pu + 1e-9))
                if n <= 0:
                    continue
                span = n * pu - gap
                s = a + (b - a - span) / 2
                st += [(s + k * pu, v) for k in range(n)]
        if st and (best_st is None or _better(st, best_st, w, h)):
            best_st = st
        # aligned: common column grid
        for phu in np.arange(0, pu, step):
            al = []
            for v, ints in rows:
                for a, b in ints:
                    k0 = math.ceil((a - umin - phu) / pu - 1e-9)
                    k1 = math.floor((b - w - umin - phu) / pu + 1e-9)
                    al += [(umin + phu + k * pu, v) for k in range(k0, k1 + 1)]
            if al and (best_al is None or _better(al, best_al, w, h)):
                best_al = al
    out = []
    if best_al:
        out.append(Layout(best_al, "", "aligned"))
    if best_st:
        out.append(Layout(best_st, "", "staggered"))
    return out


def _better(a, b, w, h):
    if len(a) != len(b):
        return len(a) > len(b)
    return _hull_area(a, w, h) < _hull_area(b, w, h) - 1e-9


def _hull_area(pts, w, h):
    p = np.array(pts)
    corners = np.vstack([p, p + [w, 0], p + [0, h], p + [w, h]])
    return shapely.MultiPoint(corners).convex_hull.area


# --------------------------------------------------------------------------- driver
def place_panels(faces, obstacles, cfg, dominant_azimuth, lat, lon, prefer="aligned"):
    sun, best_poa = relative_yield_table(lat, lon, cfg.diffuse_fraction)
    keepout = unary_union([o["geom"] for o in obstacles]) if obstacles else Polygon()
    results, panels = [], []
    for f in faces:
        poly = f.polygon
        rec = {"face": f.id, "slope": f.slope, "aspect": f.aspect, "area_plan_m2": poly.area,
               "area_true_m2": poly.area / math.cos(math.radians(f.slope)), "n_panels": 0, "status": ""}
        poa = poa_annual(f.slope, f.aspect if not f.is_flat else 180.0, sun, diffuse_fraction=cfg.diffuse_fraction)
        rec["relative_yield"] = poa / best_poa
        if f.extra.get("structure"):
            rec["status"] = "excluded: dormer / non-planar structure"
        elif f.extra.get("glazed"):
            rec["status"] = "excluded: glazing"
        elif f.slope > cfg.max_slope:
            rec["status"] = "excluded: too steep"
        elif rec["relative_yield"] < cfg.min_relative_yield:
            rec["status"] = "excluded: poor orientation (yield %.0f%%)" % (100 * rec["relative_yield"])
        if rec["status"]:
            results.append(rec)
            continue
        c = poly.centroid
        frame = PlaneFrame(f.plane, (c.x, c.y), eave_azimuth_deg=dominant_azimuth if f.is_flat else None)
        uv_face = frame.poly_to_uv(poly)
        usable = uv_face.buffer(-cfg.edge_setback, join_style="mitre")
        local_keep = keepout.intersection(poly.buffer(cfg.obstruction_clearance + 2.0))
        if not local_keep.is_empty:
            usable = usable.difference(frame.poly_to_uv(local_keep).buffer(cfg.obstruction_clearance, join_style="mitre"))
        usable = shapely.make_valid(usable)
        rec["usable_true_m2"] = usable.area
        best = None
        for orient, (w, h) in (("portrait", (cfg.panel_width, cfg.panel_length)),
                               ("landscape", (cfg.panel_length, cfg.panel_width))):
            for lay in pack_region(usable, w, h, cfg.panel_gap, cfg.offset_step):
                lay.orientation = orient
                lay.wh = (w, h)
                if best is None or lay.n > best.n or (lay.n == best.n and lay.mode == prefer and best.mode != prefer):
                    best = lay
        if best is None or best.n < cfg.min_panels_per_face:
            rec["status"] = "no room for a module"
            results.append(rec)
            continue
        w, h = best.wh
        for (u, v) in best.panels_uv:
            uv = np.array([[u, v], [u + w, v], [u + w, v + h], [u, v + h]])
            xyz = frame.to_xyz(uv)
            panels.append({"face": f.id, "orientation": best.orientation, "uv": uv, "xyz": xyz,
                           "footprint": Polygon(xyz[:, :2]), "usable_uv": usable})
        rec.update(n_panels=best.n, orientation=best.orientation, mode=best.mode,
                   kWp=best.n * cfg.panel_power_wp / 1000.0, status="used")
        f.extra["frame"] = frame
        f.extra["usable_uv"] = usable
        results.append(rec)
    return panels, results


# --------------------------------------------------------------------------- shading (DSM ray casting)
def shading_loss(panels, dsm, transform, lat, lon, n_days=24, max_dist=40.0, step=0.25):
    """Fraction of annual clear-sky beam energy lost by each panel centre to DSM occluders.

    Rays are marched from the panel centre (+5 cm) towards the sun over the full DSM
    (includes trees, dormers, chimneys and neighbouring buildings)."""
    zen, saz, D = sun_positions(lat, lon)
    sel = np.isin(D, np.linspace(1, 365, n_days).astype(int))
    zen, saz = zen[sel], saz[sel]
    dni, _ = clear_sky(zen)
    inv = ~transform
    h, w = dsm.shape
    ts = np.arange(0.5, max_dist, step)
    out = []
    for p in panels:
        cx, cy, cz = p["xyz"].mean(0)
        n = np.cross(p["xyz"][1] - p["xyz"][0], p["xyz"][3] - p["xyz"][0]); n /= np.linalg.norm(n)
        sdir = np.stack([np.sin(zen) * np.sin(saz), np.sin(zen) * np.cos(saz), np.cos(zen)], -1)
        cos_i = np.clip(sdir @ n, 0, None)
        e = dni * cos_i
        # 3-D ray: P(t) = C + t * s ; start 0.5 m out and 0.15 m above the module to avoid
        # self-occlusion by DSM noise on the module's own roof plane
        X = cx + ts[None] * sdir[:, 0:1]
        Y = cy + ts[None] * sdir[:, 1:2]
        Z = cz + 0.15 + ts[None] * sdir[:, 2:3]
        col, row = inv * (X, Y)
        col, row = np.floor(col).astype(int), np.floor(row).astype(int)
        inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        zt = np.where(inside, dsm[np.clip(row, 0, h - 1), np.clip(col, 0, w - 1)], -np.inf)
        blocked = (zt > Z).any(1)
        tot = e.sum()
        out.append(float(e[blocked].sum() / tot) if tot > 0 else 0.0)
    return np.array(out)
