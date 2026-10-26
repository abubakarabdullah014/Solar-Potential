"""Roof-face segmentation and plane regularisation.

Pipeline (all inside the building mask, on a crop around it):

1. **Normals.**  Masked Gaussian smoothing (normalised convolution, so ground
   pixels never bleed into the roof) followed by Horn derivatives.
2. **Seed quality.**  Normal dispersion ``1 - |mean(n)|`` in a small window;
   it is ~0 on planar facets and large on ridges, hips, valleys and steps.
3. **Region growing.**  Seeds are visited from most to least planar.  A
   region absorbs a 4-neighbour if (a) its normal is within ``grow_angle`` of
   the region plane normal and (b) its point-to-plane distance is below
   ``grow_dist``.  The region plane is re-estimated by least squares on
   accumulated moments whenever the region has grown by 25 %, so the test
   adapts as the facet is discovered.
4. **Merging.**  Adjacent regions whose planes differ by < ``merge_angle`` and
   whose union is fitted with RMS < ``merge_dist`` are merged (union-find).
5. **Robust refit.**  Each face plane z = a x + b y + c is re-fitted with
   iteratively re-weighted least squares (Tukey biweight), so chimneys,
   skylights and residual noise do not tilt the plane.
6. **Competitive assignment.**  Un-labelled pixels are absorbed by the
   neighbouring face whose plane explains them best (minimum |Δz|), by
   iterated frontier expansion.  Pixels no plane explains within
   ``assign_max_residual`` stay unassigned - they are obstruction candidates.
7. **Envelope rule for shared edges.**  Where two faces meet (no vertical
   step), the true boundary is their plane-plane intersection line.  For a
   convex edge (ridge / hip) the roof is the *lower* envelope
   ``min(z_A, z_B)``; for a concave edge (valley) it is the *upper* envelope
   ``max(z_A, z_B)``.  Re-labelling a band around each shared boundary with
   that rule snaps the raster boundary exactly onto the intersection line and
   makes the reconstructed surface continuous (watertight) across it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from .terrain import horn_gradient, masked_gaussian, normals_from_gradient, plane_slope_aspect


@dataclass
class Plane:
    a: float
    b: float
    c: float

    def z(self, x, y):
        return self.a * x + self.b * y + self.c

    @property
    def normal(self):
        n = np.array([-self.a, -self.b, 1.0])
        return n / np.linalg.norm(n)


@dataclass
class Face:
    id: int
    plane: Plane
    area_px: int
    rmse: float
    slope: float
    aspect: float
    mean_height: float
    polygon: object = None  # shapely polygon in map coordinates
    extra: dict = field(default_factory=dict)

    @property
    def is_flat(self):
        return self.extra.get("flat", False)


# --------------------------------------------------------------------------- planes
def fit_plane_lsq(x, y, z, w=None) -> Plane:
    A = np.column_stack([x, y, np.ones_like(x)])
    if w is None:
        sol, *_ = np.linalg.lstsq(A, z, rcond=None)
    else:
        sw = np.sqrt(w)
        sol, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
    return Plane(*map(float, sol))


def fit_plane_robust(x, y, z, iters=6, c_tukey=4.685, trim=0.6) -> tuple[Plane, np.ndarray]:
    """Robust plane fit.

    Tukey's biweight has a non-convex objective, so it needs a good start: plain LS
    is pulled towards one-sided outliers (chimneys, dormers always lie *above* the
    roof).  We therefore initialise with iterated trimmed least squares (keep the
    ``trim`` fraction of smallest residuals - a cheap LTS approximation), then
    refine with IRLS/Tukey using a MAD scale.  Returns plane and final weights."""
    p = fit_plane_lsq(x, y, z)
    if len(z) > 10:
        for _ in range(4):
            r = np.abs(z - p.z(x, y))
            keep = r <= np.quantile(r, trim)
            p = fit_plane_lsq(x[keep], y[keep], z[keep])
    w = np.ones_like(z)
    for _ in range(iters):
        r = z - p.z(x, y)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-4
        u = r / (c_tukey * s)
        w = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if w.sum() < 3:
            break
        p = fit_plane_lsq(x, y, z, w)
    return p, w


def angle_between(n1, n2) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(n1, n2)), -1, 1))))


# --------------------------------------------------------------------------- segmentation
class RoofSegmenter:
    def __init__(self, dsm, mask, res, cfg, margin=10):
        rows, cols = np.nonzero(mask)
        r0, r1 = max(rows.min() - margin, 0), min(rows.max() + margin + 1, mask.shape[0])
        c0, c1 = max(cols.min() - margin, 0), min(cols.max() + margin + 1, mask.shape[1])
        self.window = (r0, r1, c0, c1)
        self.full_shape = mask.shape
        self.z = dsm[r0:r1, c0:c1].astype(float)
        self.mask = mask[r0:r1, c0:c1]
        self.res = res
        self.cfg = cfg
        h, w = self.z.shape
        rr, cc = np.mgrid[0:h, 0:w]
        # local metric coordinates: x east, y north (origin at crop top-left pixel centre)
        self.x = cc * res
        self.y = -rr * res

    # ---- helpers
    def uncrop(self, a, fill=0):
        out = np.full(self.full_shape, fill, dtype=a.dtype)
        r0, r1, c0, c1 = self.window
        out[r0:r1, c0:c1] = a
        return out

    def to_world_plane(self, p: Plane, transform):
        """Convert crop-local plane to map coordinates z = A X + B Y + C."""
        r0, _, c0, _ = self.window
        X0 = transform.c + (c0 + 0.5) * transform.a
        Y0 = transform.f + (r0 + 0.5) * transform.e
        # x = X - X0 ; y = Y - Y0
        return Plane(p.a, p.b, p.c - p.a * X0 - p.b * Y0)

    # ---- step 1-2
    def compute_normals(self):
        zs = masked_gaussian(self.z, self.mask, self.cfg.normal_sigma)
        gx, gy = horn_gradient(zs, self.res)
        n = normals_from_gradient(gx, gy)
        k = self.cfg.planarity_window
        m = self.mask.astype(float)
        mean = np.stack([ndi.uniform_filter(n[..., i] * m, k) for i in range(3)], -1)
        cnt = ndi.uniform_filter(m, k)
        mean = mean / np.maximum(cnt, 1e-6)[..., None]
        disp = 1.0 - np.linalg.norm(mean, axis=-1)
        disp[cnt < 0.99] = 1.0  # do not seed next to the footprint boundary
        self.normals, self.zs, self.dispersion = n, zs, disp
        return n, disp

    # ---- step 3
    def region_grow(self):
        cfg = self.cfg
        h, w = self.z.shape
        labels = np.zeros((h, w), np.int32)
        cos_t = np.cos(np.radians(cfg.grow_angle))
        disp = self.dispersion
        seeds_thr = 1 - np.cos(np.radians(2.0))  # very planar
        rr, cc = np.nonzero(self.mask & (disp < seeds_thr))
        order = np.argsort(disp[rr, cc])
        z, x, y, n = self.zs, self.x, self.y, self.normals
        nxt = 1
        for k in order:
            r, c = rr[k], cc[k]
            if labels[r, c]:
                continue
            # moments for LSQ
            S = np.zeros(9)
            def add(rr_, cc_):
                xv, yv, zv = x[rr_, cc_], y[rr_, cc_], z[rr_, cc_]
                S[:] += (1, xv, yv, zv, xv * xv, xv * yv, yv * yv, xv * zv, yv * zv)
            nrm = n[r, c].copy()
            a, b = -nrm[0] / nrm[2], -nrm[1] / nrm[2]
            cc0 = z[r, c] - a * x[r, c] - b * y[r, c]
            labels[r, c] = nxt
            add(r, c)
            q = deque([(r, c)])
            size, next_fit = 1, 30
            members = [(r, c)]
            while q:
                pr, pc = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    r2, c2 = pr + dr, pc + dc
                    if r2 < 0 or c2 < 0 or r2 >= h or c2 >= w:
                        continue
                    if labels[r2, c2] or not self.mask[r2, c2]:
                        continue
                    if np.dot(n[r2, c2], nrm) < cos_t:
                        continue
                    if abs(z[r2, c2] - (a * x[r2, c2] + b * y[r2, c2] + cc0)) > cfg.grow_dist:
                        continue
                    labels[r2, c2] = nxt
                    add(r2, c2)
                    members.append((r2, c2))
                    q.append((r2, c2))
                    size += 1
                    if size >= next_fit:
                        next_fit = int(size * 1.25) + 1
                        M = np.array([[S[4], S[5], S[1]], [S[5], S[6], S[2]], [S[1], S[2], S[0]]])
                        v = np.array([S[7], S[8], S[3]])
                        try:
                            a, b, cc0 = np.linalg.solve(M, v)
                            nrm = np.array([-a, -b, 1.0]); nrm /= np.linalg.norm(nrm)
                        except np.linalg.LinAlgError:
                            pass
            if size * self.res ** 2 < cfg.min_region_area:
                for (pr, pc) in members:
                    labels[pr, pc] = -1  # visited, too small
            else:
                nxt += 1
        labels[labels < 0] = 0
        self.labels = labels
        return labels

    # ---- step 4-5
    def fit_all(self, labels):
        planes = {}
        for l in np.unique(labels):
            if l <= 0:
                continue
            m = labels == l
            p, _ = fit_plane_robust(self.x[m], self.y[m], self.z[m])
            planes[int(l)] = p
        return planes

    def adjacency(self, labels):
        pairs = set()
        for a_, b_ in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
            m = (a_ > 0) & (b_ > 0) & (a_ != b_)
            for u, v in zip(a_[m], b_[m]):
                pairs.add((min(u, v), max(u, v)))
        # also through 1-2 px unlabelled gaps (edges between regions)
        lab_d = ndi.grey_dilation(labels, size=3)
        lab_e = -ndi.grey_dilation(-np.where(labels > 0, labels, 10 ** 6), size=3)
        m = (lab_d > 0) & (lab_e < 10 ** 6) & (lab_d != lab_e) & (labels == 0)
        for u, v in zip(lab_d[m], lab_e[m]):
            if u > 0 and v > 0:
                pairs.add((min(u, v), max(u, v)))
        return pairs

    def merge(self, labels):
        cfg = self.cfg
        changed = True
        while changed:
            changed = False
            planes = self.fit_all(labels)
            cands = []
            for u, v in self.adjacency(labels):
                ang = angle_between(planes[u].normal, planes[v].normal)
                if ang < cfg.merge_angle:
                    cands.append((ang, u, v))
            cands.sort()
            used = set()
            for ang, u, v in cands:
                if u in used or v in used:
                    continue
                m = (labels == u) | (labels == v)
                p, _ = fit_plane_robust(self.x[m], self.y[m], self.z[m])
                rms = np.sqrt(np.mean((self.z[m] - p.z(self.x[m], self.y[m])) ** 2))
                if rms < cfg.merge_dist:
                    labels[labels == v] = u
                    used.update((u, v))
                    changed = True
        return labels

    # ---- step 6
    def residual_stack(self, planes):
        ids = sorted(planes)
        R = np.stack([np.abs(self.z - planes[i].z(self.x, self.y)) for i in ids])
        return ids, R

    def assign(self, labels, planes, max_res):
        ids, R = self.residual_stack(planes)
        lut = {l: k for k, l in enumerate(ids)}
        labels = labels.copy()
        for _ in range(500):
            un = self.mask & (labels == 0)
            if not un.any():
                break
            cost = np.full(R.shape, np.inf)
            for k, l in enumerate(ids):
                fr = ndi.binary_dilation(labels == l) & un
                cost[k][fr] = R[k][fr]
            best = np.argmin(cost, 0)
            bc = np.take_along_axis(cost, best[None], 0)[0]
            ok = np.isfinite(bc) & (bc < max_res)
            if not ok.any():
                break
            labels[ok] = np.array(ids)[best[ok]]
        return labels

    def reassign_competitive(self, labels, planes, iters=3):
        """Boundary pixels switch to an adjacent face if that face explains them better."""
        ids, R = self.residual_stack(planes)
        for _ in range(iters):
            cur = np.full(labels.shape, np.inf)
            for k, l in enumerate(ids):
                cur[labels == l] = R[k][labels == l]
            new = labels.copy()
            best = cur.copy()
            for k, l in enumerate(ids):
                nb = ndi.binary_dilation(labels == l) & (labels > 0) & (labels != l)
                better = nb & (R[k] < best - 0.02)
                new[better] = l
                best[better] = R[k][better]
            if np.array_equal(new, labels):
                break
            labels = new
        return labels

    # ---- step 7
    def classify_edges(self, labels, planes):
        """Return {(u,v): 'convex'|'concave'|'step'} for adjacent faces."""
        edges = {}
        for u, v in self.adjacency(labels):
            mu, mv = labels == u, labels == v
            band = (ndi.binary_dilation(mu, iterations=2) & mv) | (ndi.binary_dilation(mv, iterations=2) & mu)
            if band.sum() < 3:
                continue
            dz = planes[u].z(self.x[band], self.y[band]) - planes[v].z(self.x[band], self.y[band])
            if np.median(np.abs(dz)) > self.cfg.step_edge_height:
                edges[(u, v)] = "step"
                continue
            # Evaluate both planes in the interior of u (away from the edge)
            inner_u = mu & ~ndi.binary_dilation(mv, iterations=4) & ndi.binary_dilation(mv, iterations=12)
            if inner_u.sum() < 3:
                inner_u = mu
            d = np.median(planes[v].z(self.x[inner_u], self.y[inner_u]) - planes[u].z(self.x[inner_u], self.y[inner_u]))
            edges[(u, v)] = "convex" if d > 0 else "concave"
        return edges

    def envelope_relabel(self, labels, planes, edges):
        band_px = max(1, int(round(self.cfg.envelope_band / self.res)))
        out = labels.copy()
        for (u, v), kind in edges.items():
            if kind == "step":
                continue
            mu, mv = labels == u, labels == v
            near = (ndi.binary_dilation(mu, iterations=band_px) & mv) | (ndi.binary_dilation(mv, iterations=band_px) & mu)
            if not near.any():
                continue
            zu = planes[u].z(self.x[near], self.y[near])
            zv = planes[v].z(self.x[near], self.y[near])
            pick_u = zu <= zv if kind == "convex" else zu >= zv
            out[near] = np.where(pick_u, u, v)
        return out

    # ---- driver
    def run(self):
        cfg = self.cfg
        self.compute_normals()
        labels = self.region_grow()
        labels = self.merge(labels)
        planes = self.fit_all(labels)
        labels = self.assign(labels, planes, cfg.assign_max_residual)
        planes = self.fit_all(labels)
        labels = self.reassign_competitive(labels, planes)
        labels = self.merge(labels)
        planes = self.fit_all(labels)
        edges = self.classify_edges(labels, planes)
        labels = self.envelope_relabel(labels, planes, edges)
        labels = _drop_fragments(labels, int(cfg.min_region_area / self.res ** 2))
        labels = self.assign(labels, planes, cfg.assign_max_residual)
        planes = self.fit_all(labels)
        # relabel consecutively
        uniq = [l for l in np.unique(labels) if l > 0]
        remap = np.zeros(labels.max() + 1, np.int32)
        for i, l in enumerate(uniq, 1):
            remap[l] = i
        labels = remap[labels]
        planes = {int(remap[l]): p for l, p in planes.items() if l in uniq}
        edges = self.classify_edges(labels, planes)
        self.labels, self.planes, self.edges = labels, planes, edges
        return labels, planes, edges

    def roof_surface(self, labels, planes):
        """Regularised roof surface: each pixel takes the height of its face plane.
        Unassigned roof pixels (obstructions) take the plane of the nearest face, i.e.
        the roof *underneath* the obstruction."""
        zr = np.full(self.z.shape, np.nan)
        for l, p in planes.items():
            m = labels == l
            zr[m] = p.z(self.x[m], self.y[m])
        un = self.mask & (labels == 0)
        if un.any() and (labels > 0).any():
            _, (ir, ic) = ndi.distance_transform_edt(labels == 0, return_indices=True)
            near = labels[ir, ic]
            for l, p in planes.items():
                m = un & (near == l)
                zr[m] = p.z(self.x[m], self.y[m])
        return zr


def _drop_fragments(labels, min_px):
    """Keep only the largest connected piece of each label if others are tiny."""
    out = labels.copy()
    for l in np.unique(labels):
        if l <= 0:
            continue
        lab, n = ndi.label(labels == l)
        if n <= 1:
            if (lab > 0).sum() < min_px:
                out[labels == l] = 0
            continue
        sizes = ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
        for k, s in enumerate(sizes, 1):
            if s < min_px:
                out[lab == k] = 0
    return out


def build_faces(seg: RoofSegmenter, labels, planes, transform, flat_slope) -> list[Face]:
    faces = []
    for l, p in planes.items():
        m = labels == l
        r = seg.z[m] - p.z(seg.x[m], seg.y[m])
        wp = seg.to_world_plane(p, transform)
        slope, aspect = plane_slope_aspect(wp.a, wp.b)
        faces.append(Face(id=l, plane=wp, area_px=int(m.sum()), rmse=float(np.sqrt(np.mean(r ** 2))),
                          slope=slope, aspect=aspect, mean_height=float(seg.z[m].mean()),
                          extra={"flat": slope < flat_slope,
                                 "mae": float(np.mean(np.abs(r))),
                                 "p95_abs_residual": float(np.percentile(np.abs(r), 95))}))
    return faces
