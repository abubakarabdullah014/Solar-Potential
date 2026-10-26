# 5-minute demo outline (suggested)

| time | show | say (key points) |
|---|---|---|
| 0:00–0:30 | `01_inputs.png` | Data: 10 cm DSM + RGB, UTM 30N, ~52° N. The clipped DSM is the parcel mask. Target: an atrium-type hipped roof with dormers, chimneys and a glass lantern. |
| 0:30–1:00 | terminal: `python -m roofsolar --out outputs` | One command, ~30 s, prints metrics and the constraint check. |
| 1:00–1:30 | `02_elevation_slope_aspect.png` | Horn 3×3 derivatives; the aspect convention; outliers via MAD; harmonic inpainting (exact for planes). |
| 1:30–2:30 | `03_roof_faces_and_edges.png`, `04_regularization.png` | Region growing on normals, robust (trimmed + Tukey) plane fits, envelope rule, coverage-preserving simplification, vertex snapping to plane intersections → watertight (seam gap 0.78 m → 0). Cross-section: planes vs DSM. |
| 2:30–3:15 | `05_obstructions.png` | Two cues: DSM residual (chimneys, dormers) and Lab colour (glass is less red; shadows only darken). The lantern is detected as glazed faces; tree overhang via ExG. |
| 3:15–4:15 | `06_panel_layout.png`, code `geometry.PlaneFrame` | Yield screening; true-size modules in the plane's (u, v) frame (the cos θ correction); exact row-interval packing; shading ray-cast drops 2 modules; 48 modules / 20.4 kWp; 0 violations. |
| 4:15–4:45 | `model/roof_model_3d.html` or `turntable.mp4` | 3-D model: facets, walls, obstructions, modules. |
| 4:45–5:00 | `docs/TECHNICAL_REPORT.md` §9, §11 | Robustness table; honest limitations (trees, smoothing, footprint snapping). |
