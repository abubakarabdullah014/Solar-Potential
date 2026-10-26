"""Vector geometry: face polygons, coverage simplification, ridge/hip/valley lines,
and the affine map between map coordinates and a roof plane's own 2-D frame."""
from __future__ import annotations

import numpy as np
import shapely
from rasterio import features
from shapely.geometry import shape, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union


def label_polygons(labels_full: np.ndarray, transform, ids) -> dict[int, Polygon]:
    polys: dict[int, list] = {}
    for geom, val in features.shapes(labels_full.astype(np.int32), mask=labels_full > 0, transform=transform,
                                     connectivity=4):
        v = int(val)
        if v in ids:
            polys.setdefault(v, []).append(shape(geom))
    out = {}
    for v, ps in polys.items():
        g = unary_union(ps)
        if isinstance(g, MultiPolygon):  # keep dominant piece
            g = max(g.geoms, key=lambda p: p.area)
        out[v] = g
    return out


def simplify_coverage(polys: dict[int, Polygon], tol: float) -> dict[int, Polygon]:
    """Topology-preserving simplification: shared edges are simplified once, so
    neighbouring faces stay gap- and overlap-free."""
    ids = list(polys)
    arr = shapely.coverage_simplify(np.array([polys[i] for i in ids], dtype=object), tol)
    out = {}
    for i, g in zip(ids, arr):
        g = shapely.make_valid(g)
        if isinstance(g, MultiPolygon) or g.geom_type == "GeometryCollection":
            parts = [p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"]
            if not parts:
                continue
            g = max(parts, key=lambda p: p.area)
        if g.area > 0:
            out[i] = g
    return out


def plane_intersection_line(p1, p2):
    """2-D projection of the intersection of z=a1x+b1y+c1 and z=a2x+b2y+c2:
    (a1-a2)x + (b1-b2)y + (c1-c2) = 0.  Returns (point, direction) or None if parallel."""
    A, B, C = p1.a - p2.a, p1.b - p2.b, p1.c - p2.c
    nrm = np.hypot(A, B)
    if nrm < 1e-9:
        return None
    d = np.array([-B, A]) / nrm
    p0 = -C * np.array([A, B]) / (nrm ** 2)
    return p0, d


def shared_edge_lines(polys, planes, edges, min_len=0.5):
    """For each convex/concave adjacency, the exact 3-D ridge/hip/valley segment:
    the plane-plane intersection line clipped to the shared polygon boundary."""
    out = []
    for (u, v), kind in edges.items():
        if u not in polys or v not in polys:
            continue
        shared = polys[u].boundary.intersection(polys[v].buffer(0.15))
        if shared.is_empty or shared.length < min_len:
            continue
        if kind == "step":
            out.append({"faces": (u, v), "kind": "step", "geom2d": shared})
            continue
        res = plane_intersection_line(planes[u], planes[v])
        if res is None:
            continue
        p0, d = res
        pts = np.array(shapely.get_coordinates(shared))
        t = (pts - p0) @ d
        a, b = p0 + d * t.min(), p0 + d * t.max()
        seg = LineString([a, b])
        # classify geometric type: horizontal -> ridge (convex) / valley... sloped convex -> hip
        za, zb = planes[u].z(*a), planes[u].z(*b)
        horiz = abs(za - zb) / max(seg.length, 1e-6) < np.tan(np.radians(5))
        name = ("ridge" if horiz else "hip") if kind == "convex" else "valley"
        out.append({"faces": (u, v), "kind": name, "geom2d": seg,
                    "z": (float(za), float(zb)),
                    "fit_dist": float(np.max(np.abs((pts - p0) @ np.array([-d[1], d[0]]))))})
    return out


class PlaneFrame:
    """Affine bijection between map XY and true in-plane distances (u, v).

    u : horizontal unit vector lying in the plane (along the contour / eave)
    v : unit vector in the plane pointing up-slope
    A horizontal displacement of length L along the down-slope azimuth
    corresponds to an in-plane distance L / cos(slope) - this is the
    foreshortening that must be undone to place true-size modules.
    """

    def __init__(self, plane, origin_xy, eave_azimuth_deg=None):
        self.plane = plane
        a, b = plane.a, plane.b
        g = np.hypot(a, b)
        self.cos_s = 1.0 / np.sqrt(1.0 + g * g)
        if g > 1e-3 and eave_azimuth_deg is None:
            up = np.array([a, b]) / g          # horizontal up-slope direction (x east, y north)
        else:
            th = np.radians(eave_azimuth_deg or 0.0)  # azimuth of eave line for flat roofs
            along = np.array([np.sin(th), np.cos(th)])
            up = np.array([-along[1], along[0]])
        self.up = up
        self.along = np.array([up[1], -up[0]])  # horizontal, perpendicular to up-slope
        self.o = np.asarray(origin_xy, float)
        self.k = self.cos_s if g > 1e-3 else 1.0  # scale for v-axis
        # forward: [u, v] = M @ (xy - o)
        self.M = np.vstack([self.along, up / self.k])
        self.Minv = np.linalg.inv(self.M)

    def to_uv(self, xy):
        xy = np.atleast_2d(xy)
        return (xy - self.o) @ self.M.T

    def to_xy(self, uv):
        uv = np.atleast_2d(uv)
        return uv @ self.Minv.T + self.o

    def to_xyz(self, uv):
        xy = self.to_xy(uv)
        return np.column_stack([xy, self.plane.z(xy[:, 0], xy[:, 1])])

    def poly_to_uv(self, geom):
        return shapely.transform(geom, lambda c: self.to_uv(c))

    def poly_to_xy(self, geom):
        return shapely.transform(geom, lambda c: self.to_xy(c))


def snap_to_plane_intersections(polys: dict, planes: dict, edges: dict, max_snap=0.6):
    """Make the roof geometrically consistent (watertight across ridges/hips/valleys).

    Every polygon vertex shared by faces whose common edge is convex or concave is
    moved onto the analytic intersection of the face planes:
      * 2 faces : orthogonal projection onto the 2-D intersection line
                  (a1-a2)x + (b1-b2)y + (c1-c2) = 0
      * >=3 faces: least-squares point of all pairwise intersection lines (for
                  three planes meeting at a hip/valley apex this is their exact
                  common point)
    Moves larger than ``max_snap`` are rejected (the boundary is then not a true
    plane-plane seam).  Because the coverage shares vertices exactly, applying the
    same move to every polygon keeps the partition gap- and overlap-free.
    At a snapped vertex z_A(x,y) = z_B(x,y), so adjacent facets meet without cracks.
    """
    from shapely.strtree import STRtree
    from shapely.geometry import Point

    ids = list(polys)
    geoms = [polys[i] for i in ids]
    tree = STRtree(geoms)
    verts = {}
    for i, g in zip(ids, geoms):
        for ring in [g.exterior] + list(g.interiors):
            for c in ring.coords:
                verts.setdefault((round(c[0], 6), round(c[1], 6)), set()).add(i)
    # faces touching each vertex (a vertex of A can lie on an edge of B)
    for v in list(verts):
        for j in tree.query(Point(v).buffer(1e-6)):
            if geoms[j].boundary.distance(Point(v)) < 1e-6:
                verts[v].add(ids[j])
    good = {k for k, kind in edges.items() if kind in ("convex", "concave")}
    moves = {}
    n2 = n3 = 0
    for v, fs in verts.items():
        fs = sorted(fs)
        pairs = [(u, w) for a, u in enumerate(fs) for w in fs[a + 1:] if (min(u, w), max(u, w)) in good]
        if not pairs:
            continue
        A = np.array([[planes[u].a - planes[w].a, planes[u].b - planes[w].b] for u, w in pairs])
        c = np.array([planes[u].c - planes[w].c for u, w in pairs])
        nrm = np.linalg.norm(A, axis=1)
        keep = nrm > 1e-6
        if not keep.any():
            continue
        A, c = A[keep] / nrm[keep, None], c[keep] / nrm[keep]
        p = np.array(v)
        if len(A) == 1 or np.linalg.matrix_rank(A, tol=1e-3) < 2:
            n = A[0]
            q = p - (n @ p + c[0]) * n          # projection onto the line
            kind = 2
        else:
            q, *_ = np.linalg.lstsq(A, -c, rcond=None)
            kind = 3
        if np.linalg.norm(q - p) <= max_snap:
            moves[v] = tuple(q)
            n2 += kind == 2
            n3 += kind == 3

    def mv(coords):
        out = coords.copy()
        for k, cc in enumerate(coords):
            key = (round(cc[0], 6), round(cc[1], 6))
            if key in moves:
                out[k] = moves[key]
        return out

    out = {}
    for i, g in zip(ids, geoms):
        ng = shapely.transform(g, mv)
        if not ng.is_valid:
            ng = shapely.make_valid(ng)
            parts = [p for p in getattr(ng, "geoms", [ng]) if p.geom_type == "Polygon"]
            ng = max(parts, key=lambda p: p.area) if parts else g
        out[i] = ng
    return out, {"snapped_line": n2, "snapped_junction": n3, "candidates": len(verts)}
