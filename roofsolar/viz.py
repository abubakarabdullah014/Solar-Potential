"""Figures for the report / video."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource
from matplotlib.patches import Polygon as MplPolygon, Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

OBS_COLORS = {"chimney/vent": "#e5383b", "dormer/raised structure": "#f4a261",
              "skylight/lantern (raised glazing)": "#00b4d8", "roof window (flush)": "#70e000",
              "overhanging vegetation": "#2d6a4f"}
EDGE_COLORS = {"ridge": "#d00000", "hip": "#c77dff", "valley": "#00f5d4", "step": "#ffffff"}


def _extent(tr, shape):
    return [tr.c, tr.c + tr.a * shape[1], tr.f + tr.e * shape[0], tr.f]


def _zoom(ax, R, pad=2.0):
    from shapely.ops import unary_union
    b = unary_union([f.polygon for f in R.faces]).bounds
    ax.set_xlim(b[0] - pad, b[2] + pad)
    ax.set_ylim(b[1] - pad, b[3] + pad)
    ax.set_aspect("equal")


def _poly(ax, g, **kw):
    for p in getattr(g, "geoms", [g]):
        if p.geom_type != "Polygon":
            continue
        ax.add_patch(MplPolygon(np.asarray(p.exterior.coords)[:, :2], closed=True, **kw))
        for r in p.interiors:
            ax.add_patch(MplPolygon(np.asarray(r.coords)[:, :2], closed=True,
                                    **{**kw, "facecolor": "none"}))


def _rgb(ax, R):
    ax.imshow(R.scene.rgb, extent=_extent(R.scene.transform, R.scene.shape))


def fig_inputs(R, path):
    sc = R.scene
    ext = _extent(sc.transform, sc.shape)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))
    ax[0].imshow(sc.rgb, extent=ext); ax[0].set_title("RGB orthophoto")
    ls = LightSource(315, 40)
    hs = ls.hillshade(R.dsm_clean, dx=sc.res, dy=sc.res)
    ax[1].imshow(hs, cmap="gray", extent=ext)
    im = ax[1].imshow(R.dsm_clean, cmap="terrain", alpha=0.55, extent=ext); plt.colorbar(im, ax=ax[1], shrink=0.8, label="m")
    ax[1].contour(np.flipud(sc.parcel), [0.5], colors="k", linewidths=1, extent=ext, origin="lower")
    ax[1].set_title("DSM (hillshade) + parcel")
    im = ax[2].imshow(np.clip(R.ndsm, 0, 12), cmap="magma", extent=ext); plt.colorbar(im, ax=ax[2], shrink=0.8, label="m")
    ax[2].contour(np.flipud(R.building), [0.5], colors="c", linewidths=1.5, extent=ext, origin="lower")
    ax[2].contour(np.flipud(R.veg & (R.ndsm > 2)), [0.5], colors="lime", linewidths=0.5, extent=ext, origin="lower")
    ax[2].set_title("nDSM, building footprint (cyan), vegetation (green)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def fig_slope_aspect(R, path):
    tc = R.crop_transform
    sh = R.labels.shape
    ext = _extent(tc, sh)
    r0, r1, c0, c1 = R.seg.window
    m = R.seg.mask
    sl = np.where(m, R.slope[r0:r1, c0:c1], np.nan)
    asp = np.where(m, R.aspect[r0:r1, c0:c1], np.nan)
    z = np.where(m, R.dsm_clean[r0:r1, c0:c1], np.nan)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
    im = ax[0].imshow(z, cmap="viridis", extent=ext); plt.colorbar(im, ax=ax[0], shrink=0.8, label="elevation m")
    ax[0].set_title("Elevation (roof)")
    im = ax[1].imshow(sl, cmap="YlOrRd", vmin=0, vmax=60, extent=ext); plt.colorbar(im, ax=ax[1], shrink=0.8, label="slope °")
    ax[1].set_title("Slope (Horn 3x3)")
    im = ax[2].imshow(asp, cmap="twilight_shifted", vmin=0, vmax=360, extent=ext)
    cb = plt.colorbar(im, ax=ax[2], shrink=0.8, label="aspect ° (0=N, 90=E)"); cb.set_ticks([0, 90, 180, 270, 360])
    ax[2].set_title("Aspect / azimuth")
    for a in ax:
        _zoom(a, R, 1); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def fig_segmentation(R, path):
    fig, ax = plt.subplots(1, 2, figsize=(17, 8.5))
    rng = np.random.default_rng(3)
    cols = {f.id: rng.uniform(0.2, 0.95, 3) for f in R.faces}
    _rgb(ax[0], R)
    for f in R.faces:
        _poly(ax[0], f.polygon, facecolor=(*cols[f.id], 0.45), edgecolor="yellow", lw=1)
        c = f.polygon.representative_point()
        ax[0].text(c.x, c.y, str(f.id), color="white", fontsize=8, ha="center", weight="bold")
    ax[0].set_title(f"Roof faces ({len(R.faces)} planes)")
    _rgb(ax[1], R)
    for e in R.edge_lines:
        g = e["geom2d"]
        for gg in getattr(g, "geoms", [g]):
            if gg.geom_type == "LineString":
                x, y = gg.xy
                ax[1].plot(x, y, color=EDGE_COLORS[e["kind"]], lw=2.2 if e["kind"] != "step" else 1.2,
                           ls="-" if e["kind"] != "step" else "--")
    for f in R.faces:
        c = f.polygon.representative_point()
        L = 1.2
        dx, dy = np.sin(np.radians(f.aspect)) * L, np.cos(np.radians(f.aspect)) * L
        if not f.is_flat:
            ax[1].annotate("", (c.x + dx, c.y + dy), (c.x, c.y), arrowprops=dict(arrowstyle="->", color="yellow", lw=1.4))
        ax[1].text(c.x, c.y - 0.6, f"{f.slope:.0f}°", color="yellow", fontsize=7, ha="center")
    ax[1].legend(handles=[Line2D([], [], color=c, lw=2, label=k) for k, c in EDGE_COLORS.items()] +
                 [Line2D([], [], color="yellow", marker=r"$\rightarrow$", lw=0, label="down-slope (aspect)")],
                 loc="lower left", fontsize=8, facecolor="0.3", labelcolor="w")
    ax[1].set_title("Ridges / hips / valleys = plane-plane intersections")
    for a in ax:
        _zoom(a, R); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def fig_regularization(R, path, roof_planes):
    tc = R.crop_transform
    sh = R.labels.shape
    ext = _extent(tc, sh)
    r0, r1, c0, c1 = R.seg.window
    m = R.seg.mask
    z = np.where(m, R.scene.dsm[r0:r1, c0:c1], np.nan)
    zr = np.where(m, roof_planes[r0:r1, c0:c1], np.nan)
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
    ls = LightSource(315, 40)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax[0].imshow(ls.hillshade(np.nan_to_num(z, nan=np.nanmin(z)), dx=.1, dy=.1), cmap="gray", extent=ext)
    ax[0].set_title("Input DSM (hillshade)")
    ax[1].imshow(ls.hillshade(np.nan_to_num(zr, nan=np.nanmin(zr)), dx=.1, dy=.1), cmap="gray", extent=ext)
    ax[1].set_title("Regularised roof: piece-wise planar")
    im = ax[2].imshow(z - zr, cmap="RdBu_r", vmin=-0.5, vmax=0.5, extent=ext)
    plt.colorbar(im, ax=ax[2], shrink=0.8, label="DSM − plane (m)")
    ax[2].set_title("Residual (red = above roof plane → obstruction)")
    for a in ax:
        _zoom(a, R, 1); a.set_xticks([]); a.set_yticks([])
    # profile along the building's principal diagonal
    ax2 = fig.add_subplot(gs[1, :])
    rr, cc = np.nonzero(m)
    th = np.radians(R.dominant_azimuth)
    d = np.array([np.sin(th), -np.cos(th)])  # (col, row) direction across the main ridge
    cr, ccen = rr.mean(), cc.mean()
    t = np.arange(-200, 200)
    pr, pc = (cr + t * d[1]).astype(int), (ccen + t * d[0]).astype(int)
    ok = (pr >= 0) & (pr < sh[0]) & (pc >= 0) & (pc < sh[1])
    pr, pc, t = pr[ok], pc[ok], t[ok]
    raw = R.scene.dsm[r0:r1, c0:c1][pr, pc]
    ax2.plot(t * R.scene.res, raw, color="0.5", lw=1.2, label="input DSM")
    ax2.plot(t * R.scene.res, zr[pr, pc], color="C3", lw=2, label="reconstructed planes")
    ax2.set_xlabel("distance along section (m)"); ax2.set_ylabel("height (m)"); ax2.legend()
    ax2.set_title("Cross-section through the building centre (perpendicular to main axis)")
    ax2.grid(alpha=0.3)
    xs, ys = tc * (pc, pr)
    ax[1].plot(xs, ys, "c-", lw=1)
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def fig_obstructions(R, path):
    fig, ax = plt.subplots(figsize=(10, 10))
    _rgb(ax, R)
    for f in R.faces:
        _poly(ax, f.polygon, facecolor="none", edgecolor="yellow", lw=0.6, alpha=0.7)
    for o in R.obstacles:
        _poly(ax, o["geom"], facecolor=(*matplotlib.colors.to_rgb(OBS_COLORS[o["kind"]]), 0.35),
              edgecolor=OBS_COLORS[o["kind"]], lw=2)
        c = o["geom"].representative_point()
        ax.text(c.x, c.y, f'{o["height"]:.1f}m' if o["kind"] != "roof window (flush)" else "", color="w", fontsize=7,
                ha="center")
    kinds = sorted({o["kind"] for o in R.obstacles})
    ax.legend(handles=[Patch(facecolor=OBS_COLORS[k], label=k) for k in kinds], loc="lower left", fontsize=9)
    ax.set_title("Detected obstructions (label = height above roof plane)")
    _zoom(ax, R); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def fig_panels(R, path):
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    summ = {s["face"]: s for s in R.face_summary}
    for a in ax:
        _rgb(a, R)
    for f in R.faces:
        st = summ[f.id]["status"]
        col = "#2a9d8f" if st == "used" else ("#e76f51" if "orientation" in st else "#999999")
        _poly(ax[0], f.polygon, facecolor=(*matplotlib.colors.to_rgb(col), 0.35), edgecolor="w", lw=0.6)
        if f.extra.get("usable_uv") is not None:
            _poly(ax[0], f.extra["frame"].poly_to_xy(f.extra["usable_uv"]), facecolor="none", edgecolor="yellow",
                  lw=1, ls="--")
    for o in R.obstacles:
        _poly(ax[0], o["geom"], facecolor="none", edgecolor=OBS_COLORS[o["kind"]], lw=1.5)
    ax[0].legend(handles=[Patch(facecolor="#2a9d8f", label="suitable face"),
                          Patch(facecolor="#e76f51", label=f"poor orientation (<{100 * R.cfg.solar.min_relative_yield:.0f}% of optimum yield)"),
                          Patch(facecolor="#999999", label="glazing / dormer / too small"),
                          Line2D([], [], color="yellow", ls="--", label="usable area (setback + keep-outs)")],
                 loc="lower left", fontsize=8)
    ax[0].set_title("Suitability and usable area")
    cmap = plt.get_cmap("RdYlGn_r")
    for p in R.panels:
        sl = p.get("shading_loss", 0)
        _poly(ax[1], p["footprint"], facecolor=cmap(min(sl / 0.4, 1)), edgecolor="k", lw=0.5, alpha=0.95)
    for o in R.obstacles:
        _poly(ax[1], o["geom"], facecolor="none", edgecolor=OBS_COLORS[o["kind"]], lw=1.2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 40)); sm.set_array([])
    plt.colorbar(sm, ax=ax[1], shrink=0.7, label="annual beam shading loss (%)")
    kwp = len(R.panels) * R.cfg.solar.panel_power_wp / 1000
    ax[1].set_title(f"PV layout: {len(R.panels)} modules × {R.cfg.solar.panel_power_wp:.0f} Wp = {kwp:.1f} kWp")
    for a in ax:
        _zoom(a, R); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def fig_3d(mb, path, elev=38, azim=-120, title="3-D reconstruction"):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    allv = []
    light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
    for name, color, m in mb.parts:
        v, f = m.vertices, m.faces
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = 0.55 + 0.45 * np.abs(n @ light)
        base = np.array(color[:3]) / 255
        fc = np.clip(base[None] * shade[:, None], 0, 1)
        fc = np.column_stack([fc, np.full(len(fc), color[3] / 255)])
        ax.add_collection3d(Poly3DCollection(tri, facecolors=fc, edgecolors=(0, 0, 0, 0.08) if "pv" not in name else (1, 1, 1, .5), linewidths=0.3))
        allv.append(v)
    V = np.vstack(allv)
    mn, mx = V.min(0), V.max(0)
    ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
    ax.set_box_aspect(mx - mn)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)
