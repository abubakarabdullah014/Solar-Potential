"""3-D model of the reconstructed roof (LoD2-style): planar roof facets, vertical
walls at eaves and at step edges, extruded obstruction volumes and PV modules.

Facets are triangulated with ear-clipping (supports holes, e.g. a face with a
dormer cut-out); every vertex is lifted onto its face plane, so each facet is
exactly planar.  Walls: an edge of face A gets a vertical quad down to the
neighbouring face (step edge) or to the ground (eave/verge), added only by the
higher side so walls are not duplicated.
"""
from __future__ import annotations

import json

import mapbox_earcut as earcut
import numpy as np
import shapely
import trimesh
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree


def _triangulate(poly: Polygon):
    rings = [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in poly.interiors]
    verts = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tri = earcut.triangulate_float64(verts, ends).reshape(-1, 3)
    return verts, tri


class MeshBuilder:
    def __init__(self):
        self.parts = []  # (name, color rgba, trimesh)

    def add(self, name, mesh, color):
        mesh.visual.face_colors = np.tile(np.array(color, np.uint8), (len(mesh.faces), 1))
        self.parts.append((name, color, mesh))

    def scene(self):
        sc = trimesh.Scene()
        for name, _, m in self.parts:
            sc.add_geometry(m, node_name=name, geom_name=name)
        return sc

    def export(self, path):
        self.scene().export(path)


def roof_facets(faces, origin):
    """Return list of (face_id, vertices Nx3, triangles Mx3)."""
    out = []
    ox, oy = origin
    for f in faces:
        polys = [f.polygon] if f.polygon.geom_type == "Polygon" else list(f.polygon.geoms)
        for p in polys:
            v2, tri = _triangulate(p)
            z = f.plane.z(v2[:, 0], v2[:, 1])
            out.append((f.id, np.column_stack([v2[:, 0] - ox, v2[:, 1] - oy, z]), tri))
    return out


def walls(faces, ground_z, origin, min_h=0.03):
    ox, oy = origin
    polys = [f.polygon for f in faces]
    tree = STRtree(polys)
    V, T = [], []
    for i, f in enumerate(faces):
        rings = [f.polygon.exterior] + list(f.polygon.interiors)
        for ring in rings:
            c = np.asarray(ring.coords)
            for p, q in zip(c[:-1], c[1:]):
                m = (p + q) / 2
                za_p, za_q, za_m = f.plane.z(*p), f.plane.z(*q), f.plane.z(*m)
                # neighbour across this edge
                nb = None
                for j in tree.query(Point(m).buffer(0.08)):
                    if j != i and polys[j].distance(Point(m)) < 0.08:
                        nb = faces[j]
                        break
                if nb is None:
                    zb_p = zb_q = zb_m = ground_z
                else:
                    zb_p, zb_q, zb_m = nb.plane.z(*p), nb.plane.z(*q), nb.plane.z(*m)
                if za_m - zb_m < min_h:  # only the higher side builds the wall
                    continue
                k = sum(len(v) for v in V)
                V.append(np.array([[p[0] - ox, p[1] - oy, za_p], [q[0] - ox, q[1] - oy, za_q],
                                   [q[0] - ox, q[1] - oy, zb_q], [p[0] - ox, p[1] - oy, zb_p]]))
                T.append(np.array([[k, k + 1, k + 2], [k, k + 2, k + 3]]))
    if not V:
        return None
    return trimesh.Trimesh(np.vstack(V), np.vstack(T), process=False)


def prism(poly: Polygon, z0, z1, origin):
    ox, oy = origin
    p = shapely.affinity.translate(poly, -ox, -oy)
    m = trimesh.creation.extrude_polygon(p, height=max(z1 - z0, 0.05))
    m.apply_translation([0, 0, z0])
    return m


def drape(poly, faces, origin, lift=0.05):
    """Triangulate a polygon and lift every vertex onto the roof face below it (+lift)."""
    ox, oy = origin
    poly = poly.segmentize(0.25)
    try:
        v2, tri = _triangulate(poly)
    except Exception:
        return None
    z = np.full(len(v2), np.nan)
    for f in faces:
        inside = shapely.contains_xy(f.polygon.buffer(0.05), v2[:, 0], v2[:, 1]) & np.isnan(z)
        z[inside] = f.plane.z(v2[inside, 0], v2[inside, 1])
    if np.isnan(z).all():
        return None
    z[np.isnan(z)] = np.nanmean(z)
    return trimesh.Trimesh(np.column_stack([v2[:, 0] - ox, v2[:, 1] - oy, z + lift]), tri, process=False)


def panel_mesh(panels, origin, lift=0.04):
    ox, oy = origin
    V, T = [], []
    for k, p in enumerate(panels):
        xyz = p["xyz"].copy()
        n = np.cross(xyz[1] - xyz[0], xyz[3] - xyz[0]); n /= np.linalg.norm(n)
        xyz = xyz + n * lift
        xyz[:, 0] -= ox; xyz[:, 1] -= oy
        V.append(xyz)
        b = 4 * k
        T.append([[b, b + 1, b + 2], [b, b + 2, b + 3]])
    if not V:
        return None
    return trimesh.Trimesh(np.vstack(V), np.vstack(T).reshape(-1, 3), process=False)


def build_model(faces, obstacles, panels, ground_z, roof_z_at, origin):
    mb = MeshBuilder()
    glazed = {f.id for f in faces if f.extra.get("glazed")}
    for fid, v, tri in roof_facets(faces, origin):
        col = (120, 210, 235, 255) if fid in glazed else (170, 120, 105, 255)
        mb.add(f"roof_face_{fid}" + ("_glazing" if fid in glazed else ""), trimesh.Trimesh(v, tri, process=False), col)
    glazed_union = shapely.union_all([f.polygon for f in faces if f.id in glazed]) if glazed else None
    w = walls(faces, ground_z, origin)
    if w is not None:
        mb.add("walls", w, (225, 220, 210, 255))
    colors = {"chimney/vent": (200, 60, 50, 255), "dormer/raised structure": (230, 150, 40, 255),
              "skylight/lantern (raised glazing)": (90, 200, 230, 200), "roof window (flush)": (60, 220, 120, 230),
              "overhanging vegetation": (40, 140, 60, 160)}
    for o in obstacles:
        g = o["geom"]
        for part in getattr(g, "geoms", [g]):
            if part.geom_type != "Polygon" or part.area < 0.01:
                continue
            if o["kind"] in ("roof window (flush)", "skylight/lantern (raised glazing)"):
                if glazed_union is not None and part.intersection(glazed_union).area > 0.3 * part.area:
                    continue  # already represented by the glazed roof facets
                d = drape(part, faces, origin, lift=0.06)
                if d is not None:
                    mb.add(f"obstruction_{o['id']}", d, colors[o["kind"]])
                    continue
            z_base, z_top = roof_z_at(part)   # lowest roof-plane height under it, measured DSM top
            try:
                mb.add(f"obstruction_{o['id']}", prism(part.simplify(0.05), z_base, max(z_top, z_base + 0.08), origin),
                       colors.get(o["kind"], (200, 0, 200, 255)))
            except Exception:
                pass
    pm = panel_mesh(panels, origin)
    if pm is not None:
        mb.add("pv_modules", pm, (25, 45, 110, 255))
    return mb


def plotly_figure(mb: MeshBuilder, title="Reconstructed roof", camera=None):
    import plotly.graph_objects as go
    data = []
    light = np.array([-0.35, -0.55, 0.76]); light /= np.linalg.norm(light)
    for name, color, m in mb.parts:
        v, f = m.vertices, m.faces
        tri = v[f]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = 0.45 + 0.55 * np.abs(n @ light)          # baked Lambert shading -> slopes readable
        fc = np.clip(np.array(color[:3])[None] * shade[:, None], 0, 255).astype(int)
        data.append(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                              facecolor=["rgb(%d,%d,%d)" % tuple(c) for c in fc], opacity=color[3] / 255,
                              name=name, flatshading=True, hoverinfo="name",
                              lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, fresnel=0.0)))
    fig = go.Figure(data)
    fig.update_layout(title=title, scene=dict(aspectmode="data", xaxis_title="E (m)", yaxis_title="N (m)",
                                              zaxis_title="H (m)"), margin=dict(l=0, r=0, t=40, b=0),
                      showlegend=False)
    if camera:
        fig.update_layout(scene_camera=camera)
    return fig


def plotly_html(mb: MeshBuilder, path, title="Reconstructed roof"):
    plotly_figure(mb, title).write_html(path, include_plotlyjs="cdn")


def render_png(mb: MeshBuilder, path, camera=None, title="", size=(1400, 1000)):
    """Static render with a real z-buffer: Plotly/WebGL in headless Chromium (Playwright)."""
    import plotly.io as pio
    from playwright.sync_api import sync_playwright
    fig = plotly_figure(mb, title, camera)
    fig.update_layout(width=size[0], height=size[1], paper_bgcolor="white",
                      scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    import tempfile, os
    html = pio.to_html(fig, include_plotlyjs=True, full_html=True)
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(html)
        tmp = fh.name
    try:
        with sync_playwright() as p:
            b = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            pg = b.new_page(viewport={"width": size[0], "height": size[1]})
            errs = []
            pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
            pg.goto("file://" + tmp)
            pg.wait_for_selector(".gl-container canvas", timeout=20000)
            pg.wait_for_timeout(3000)
            pg.screenshot(path=str(path))
            b.close()
            if errs:
                print("render warnings:", errs[:3])
    finally:
        os.unlink(tmp)


def render_turntable(mb: MeshBuilder, mp4_path, n_frames=90, size=(1280, 900), radius=1.35, z=0.6, fps=30):
    """Rotating-camera MP4 of the model (headless Chromium + ffmpeg) - for demos."""
    import os
    import subprocess
    import tempfile
    import plotly.io as pio
    from playwright.sync_api import sync_playwright
    fig = plotly_figure(mb, "", dict(eye=dict(x=radius, y=0, z=z)))
    fig.update_layout(width=size[0], height=size[1], paper_bgcolor="white",
                      scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    html = pio.to_html(fig, include_plotlyjs=True, full_html=True, div_id="fig")
    with tempfile.TemporaryDirectory() as td:
        page = os.path.join(td, "m.html")
        open(page, "w").write(html)
        with sync_playwright() as p:
            b = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            pg = b.new_page(viewport={"width": size[0], "height": size[1]})
            pg.goto("file://" + page)
            pg.wait_for_selector(".gl-container canvas", timeout=20000)
            pg.wait_for_timeout(2000)
            for k in range(n_frames):
                a = 2 * np.pi * k / n_frames
                pg.evaluate(f"Plotly.relayout('fig', {{'scene.camera.eye': {{x:{radius*np.cos(a)}, y:{radius*np.sin(a)}, z:{z}}}}})")
                pg.wait_for_timeout(120)
                pg.screenshot(path=os.path.join(td, f"f{k:04d}.png"))
            b.close()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(td, "f%04d.png"),
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", "-c:v", "libx264",
                        str(mp4_path)], check=True)
