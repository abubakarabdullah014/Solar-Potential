# Technical report — 3-D roof reconstruction & solar panel placement

## 0. Data and what it shows

| raster | grid | notes |
|---|---|---|
| `dsm_full.tif` | 797 × 802, 0.10 m, EPSG:32630 (UTM 30N) | float32 heights, no nodata; photogrammetric (smooth edges) |
| `dsm_clipped.tif` | 757 × 623, same grid offset by (40, 36) px | same heights, nodata outside the parcel → used as **AOI mask** |
| `rgb.tif` | identical grid to full DSM | 8-bit RGB orthophoto, well co-registered with the DSM (checked by contour overlay) |

Site ≈ 52.285° N, 0.940° W (England). The target building is a large house (~480 m²)
with an **atrium-type hipped roof**: an outer ring of hipped faces (~35–39°), inner
faces sloping *down* into a central well that holds a **glazed pyramidal lantern**;
a main NW–SE ridge with **dormers and chimneys** on the NE side, a lower NW wing
partially under **tree canopy**, and several **flush roof windows**.

## 1. Pipeline overview

```
DSM ─► outliers (MAD) ─► harmonic inpainting ─► median 3×3 ─► Horn slope/aspect
                                                     │
RGB ─► ExG (Otsu) vegetation ──► DTM (grey opening) ─► nDSM ─► building footprint
                                                     │
            normals (masked Gaussian) ─► region growing ─► merge ─► robust planes
            ─► competitive assignment ─► edge classification ─► envelope relabel
            ─► polygons ─► coverage simplify ─► vertex snapping to plane intersections
                                                     │
 residual DSM−plane  +  Lab colour anomalies ─► obstructions (classified)
                                                     │
 per face: relative yield ─► PlaneFrame (u,v) ─► setback & keep-outs ─► exact packing
 ─► DSM ray-cast shading ─► constraint validation ─► 3-D mesh / GeoJSON / figures
```

## 2. Elevation, slope, aspect

Horn's 3×3 operator (as in GDAL/ArcGIS), on the cleaned DSM:

```
dz/dx = [(c + 2f + i) − (a + 2d + g)] / 8Δ ,   dz/dy (north) = [(a + 2b + c) − (g + 2h + i)] / 8Δ
slope  = atan(√(zx² + zy²))
aspect = atan2(−zx, −zy)  mod 360°      (compass azimuth of steepest descent: 0 N, 90 E, 180 S)
normal = (−zx, −zy, 1)/‖·‖
```

Raster rows increase southwards, so the "north" kernel is flipped relative to the
row index. Unit tests check slope and aspect exactly on analytic planes in all four
directions. Per face, slope and aspect come from the fitted plane `z = ax + by + c`
using the same formulae with (a, b), not from averaging pixels, which avoids
angle-wrapping and noise.

## 3. Noise and missing data

* **Outliers**: |z − median₅(z)| > max(6·MAD₅, 0.5 m), *and* only small blobs
  (≤ 25 px). Large coherent deviations are real structures and are kept. Outliers
  are turned into missing data, so there is one code path for every bad pixel
  (374 px flagged on this DSM, mostly in tree crowns).
* **Missing data**: harmonic inpainting. Solve ∇²z = 0 on the missing pixels,
  with valid neighbours as Dirichlet boundary conditions (sparse direct solve).
  It is the minimum-membrane-energy surface, cannot overshoot (maximum principle),
  and is **exact for planes** because linear functions are harmonic. A unit test
  verifies this to 1e-8 with 20 % random drop-out plus a large hole.
* **Final regularisation**: after segmentation, every roof pixel is replaced by
  its face plane. So a hole inside a face is reconstructed by the plane fitted to
  the rest of that face, which is the geometrically correct model for a planar
  facet. Pixels under an obstruction get the plane of the nearest face, i.e. the
  roof *beneath* it.

## 4. Building footprint

* **DTM**: grey-scale morphological opening with a 12 m-radius disk, larger than
  half the building width, on a 4× decimated grid, then smoothed. **nDSM = DSM − DTM.**
* **Vegetation**: Excess Green ExG = (2G − R − B)/(R + G + B). The threshold is
  chosen by Otsu on elevated pixels (0.021 here).
* **Building** = largest connected component of (nDSM > 2.5 m) ∧ ¬vegetation,
  after a 0.3 m opening, that overlaps the parcel mask. Holes are filled.

## 5. Roof-face segmentation (computational geometry core)

1. **Normals** from a *masked* Gaussian (normalised convolution: ground pixels
   never bleed into the roof edge) followed by Horn.
2. **Planarity** = normal dispersion 1 − |mean(n)| over 5×5. Seeds are taken in
   order of increasing dispersion, so growth starts in face interiors.
3. **Region growing** (4-connected BFS). A neighbour is accepted if its normal is
   within 12° of the *current region plane* and its point-to-plane distance is
   < 0.12 m. The plane is re-solved from accumulated moments (Σx, Σxy, …,
   closed-form 3×3 normal equations) every time the region grows by 25 %, so the
   test adapts as the facet is discovered. Regions < 0.6 m² are dissolved.
4. **Merging** (greedy, union-find style): adjacent regions whose normals differ
   by < 6° and whose union fits a single plane with RMS < 0.10 m.
5. **Robust plane fit**: iterated trimmed least squares (an LTS approximation)
   to initialise, then IRLS with Tukey's biweight and MAD scale. Trimmed init
   matters because chimneys and dormers are *one-sided* outliers (always above
   the roof). A plain-LS start drags Tukey to a biased fixed point (+0.30 m
   offset in the unit test before the fix).
6. **Competitive assignment**: unlabelled pixels are absorbed by frontier
   expansion into the adjacent face with the smallest |Δz|, up to 0.20 m. Then
   boundary pixels switch faces if a neighbour plane explains them ≥ 2 cm better.
   Pixels that no plane explains stay unassigned (obstruction candidates).
7. **Edge classification** for each adjacent pair (A, B):
   * **step** if the planes differ by > 0.35 m along the shared boundary, or if
     the analytic intersection line lies > 0.6 m from the observed boundary;
   * otherwise **convex** (ridge/hip) if plane B extended over A's interior lies
     *above* A, else **concave** (valley).
8. **Envelope rule**: in a 0.8 m band around each non-step boundary, relabel each
   pixel with argmin(z_A, z_B) for convex edges (the roof is the lower envelope)
   or argmax for concave ones. This moves the raster boundary exactly onto the
   plane-plane intersection line.
9. **Vectorisation**: label raster → polygons → `shapely.coverage_simplify`
   (0.20 m). This simplification is topology-preserving: each shared edge is
   simplified once, so the partition stays gap- and overlap-free.
10. **Vertex snapping (watertightness)**: each vertex shared by faces with a
    convex/concave seam is moved onto the analytic intersection.
    * Two faces: orthogonal projection onto the 2-D line
      (a₁−a₂)x + (b₁−b₂)y + (c₁−c₂) = 0.
    * Three or more faces: the least-squares point of all pairwise lines. For
      three planes at a hip apex this is their exact common point.

    Moves > 0.6 m are rejected. The same move is applied to every polygon that
    shares the vertex, so the coverage stays consistent, and z_A = z_B at every
    seam vertex. **Seam gap p95 dropped from 0.78 m to 2·10⁻⁹ m** (167 line
    snaps, 18 junction snaps).
11. Ridges, hips and valleys are reported as the intersection segments.
    Horizontal convex edges are ridges, sloped convex edges are hips, and concave
    edges are valleys.

## 6. Obstructions

| cue | detects | rule |
|---|---|---|
| DSM residual r = z − z_plane > 0.20 m (area ≥ 0.04 m²) | chimneys, vents, dormers, lantern | < 2.5 m² → chimney/vent, else dormer/raised |
| non-planar face (p95 residual > 0.35 m) or near-vertical small face | dormer bodies/cheeks | whole face = structure, never used for PV |
| roof-wide Δa* < −6 vs roof median (CIE-Lab) | glazing (sky reflection is less red than tiles) | a face that is > 40 % glass is a *glazed face* (the lantern facets) |
| per-face Lab robust z-score, ΔE > 6 | windows, metal | glass if Δa* < −2.5, ΔL* > 12 or Δb* < −6; **shadow** if ΔL* < −12 with unchanged a* |
| per-face chromatic shift in the a*b* plane (z > 3, > 6 units, ΔL* > −8) | low-contrast flush windows | illumination-insensitive; the lightness gate rejects shadows |
| ExG in a 0.5 m ring around a tall DSM protrusion | tree canopy overhanging the roof | relabelled "overhanging vegetation" |

Detections of the same object are merged. The glass label wins only if glass
covers most of the merged footprint, so a dormer with a small window stays a
dormer. Each obstruction keeps its height above the roof plane and its source cue.

## 7. Panel placement

**Face screening.**
* Excluded: dormers/structures, glazed faces, faces steeper than 60°, and faces
  with **relative annual yield** < 75 %.
* Relative yield = plane-of-array irradiation ÷ the best tilt/azimuth at this
  latitude.
  * Beam part: hourly NOAA sun positions for a full year, Meinel clear-sky DNI,
    and exact cos(incidence).
  * Diffuse part: isotropic sky view factor (1 + cos β)/2, plus albedo 0.2.
  * The two are mixed with the UK annual diffuse share (0.5).
* This gives E/W 35° ≈ 81 %, flat ≈ 88 %, N 35° ≈ 60 %, which is consistent with
  published UK orientation tables.

**PlaneFrame — the key geometric step.** For a plane z = ax + by + c with slope θ:

* **u** = horizontal unit vector along the contour (eave direction)
* **v** = up-slope unit vector *in the plane*

A horizontal displacement L along the fall line is an in-plane distance L / cos θ, so

```
[u, v]ᵀ = [ along ; up/cosθ ] (xy − o)        (affine, exactly invertible)
```

Face polygons, setbacks (0.30 m) and obstruction keep-outs (footprints buffered
by 0.30 m) are all mapped into (u, v) and handled in **true metres on the
plane**. Modules (1.722 × 1.134 m, 2 cm gap) are placed in (u, v). Mapping back
gives 3-D corners exactly on the plane, and a map footprint that is correctly
foreshortened.

**Packing (exact per row).**
* Rows are bands of height h stacked up-slope. The row phase φv is searched on a
  5 cm grid.
* In each band, the set of u where the *whole band* lies inside the usable
  region is computed exactly. Every connected component of (band − usable)
  projects onto the u-axis as an interval (the projection of a connected set is
  connected). The free set is the complement of the union of those intervals.
* Two column strategies are tried:
  * **aligned**: a common column grid, with its phase φu also searched (a tidy
    rectangular array);
  * **staggered**: modules packed contiguously and centred in each free interval
    (more modules on hip trapezoids).
* Portrait and landscape are both evaluated. The layout with the most modules
  wins; ties go to aligned, then to the smaller convex-hull area (compactness).

**Shading.** For each module centre, rays are cast through the *full* DSM
(trees, dormers, neighbours) towards 24 days × hourly sun positions. The
annual clear-sky beam energy blocked is reported per module. Modules losing
> 30 % are dropped (2 modules under the NW trees).

**Validation (hard constraints, computed for every module):**
* edge lengths equal the module spec to 1 mm (this checks the slope correction);
* all 4 corners lie on the plane (< 1 µm);
* each module is inside the usable region;
* distance to the face edge ≥ setback, and distance to every obstruction ≥
  clearance, both measured in the roof plane;
* no overlaps (STRtree).

Result: **48 modules, 20.4 kWp, 0 violations.**

## 8. Results

| face | slope | aspect | true area m² | usable m² | yield | modules | layout |
|---|---|---|---|---|---|---|---|
| 2 (SW outer) | 35.1° | 225° | 56.3 | 45.3 | 95 % | 14 | landscape, aligned |
| 3 (SE outer) | 36.2° | 133° | 73.9 | 58.1 | 94 % | 19 | landscape, aligned |
| 6 (SW main) | 35.2° | 223° | 80.4 | 51.8 | 95 % | 11 | landscape, staggered |
| 13 (SE inner) | 36.6° | 132° | 22.5 | 14.4 | 94 % | 4 | portrait, aligned |

NE- and NW-facing faces (yield 48–66 %) are excluded. The lantern facets are
excluded as glazing, face 8 as tree-shaded, and dormers as structures. Exact
per-face numbers are in `outputs/face_summary.csv`.

Reconstruction quality:
* roof MAE 5.8 cm; 89 % of pixels within 10 cm and 97 % within 20 cm;
* RMSE 0.15 m, dominated by the smoothed eave and wall pixels of the
  photogrammetric DSM and by tree overlap;
* the synthetic hip-roof test recovers slope and aspect within 0.5° of ground truth.

## 9. Robustness (experiments/robustness.py, 3 seeds each)

| corruption | normal err med / p90 | slope err | aspect err | surface vs clean: median \|Δz\| / within 10 cm | modules |
|---|---|---|---|---|---|
| σ = 5 cm noise | 0.30° / 1.28° | 0.22° | 0.12° | 0.002 m / 95 % | 49.3 |
| σ = 10 cm noise | 0.66° / 2.33° | 0.60° | 0.32° | 0.007 m / 87 % | 47.0 |
| 20 % random drop-out | 0.03° / 0.31° | 0.03° | 0.01° | 0.000 m / 99 % | 49.3 |
| six 2 × 2 m holes on the roof | 0.07° / 0.62° | 0.05° | 0.05° | 0.001 m / 97 % | 44.7 |
| 1 % ±3 m spikes | 0.00° / 0.04° | 0.00° | 0.00° | 0.000 m / 99.7 % | 49.3 |
| all combined | 0.36° / 2.21° | 0.25° | 0.25° | 0.005 m / 89 % | 46.3 |

(Reference: 50 modules without the shading filter.) The plane orientations, which
drive both yield and layout, are essentially unaffected. Strong noise mostly
costs the small faces.

## 10. Assumptions

* The DSM and RGB are co-registered on the same grid (verified visually). The
  clipped DSM defines the parcel.
* Roof facets are planar. Every module lies flush on its facet (no tilt frames).
* The module is 1.722 × 1.134 m, 425 Wp, with 2 cm gaps, a 0.30 m setback from
  every face edge (eave, ridge, hip, valley, verge) and 0.30 m clearance around
  obstructions. All are configurable; local fire or building codes may require more.
* Ground height comes from the morphological DTM. Wall heights in the 3-D model
  run from eave to that DTM.

## 11. Limitations and incomplete items

* **Tree occlusion (NW wing)**: the DSM records canopy, not roof, there. The
  footprint is trimmed and those faces are small or noisy. They are excluded for
  orientation/shading anyway, but the geometry there is approximate.
* **Photogrammetric smoothing**: eaves and vertical walls are ramps in the DSM.
  This inflates RMSE and leaves a few unassigned pixels at dormer walls (small
  holes in the mesh).
* **Footprint outline** is simplified but not snapped to the building's
  principal axes. Rectilinear regularisation of the outer boundary would be the
  next step.
* **Dormers** are modelled as extruded prisms (LoD1.5), not as their own small
  gable roofs.
* **Obstruction colour thresholds** were tuned on this one image. A small roof
  vent near the SE ridge is not detected (no modules are placed near it). Flush
  windows are found only when they contrast in colour.
* **Yield model** is relative and clear-sky geometric with a climatological
  diffuse mix, not a full PVGIS/TMY simulation. Shading is beam-only, at the
  module centre, with no inter-row or electrical (string) effects.
* **Layout** is per face. It does not optimise string/inverter design or mixed
  portrait/landscape within a face.

## 12. Parameter rationale (short)

* `grow_angle` 12° / `grow_dist` 0.12 m: above the normal noise (≈ 2–3° after
  σ = 1.5 px smoothing), below the smallest real crease (a dormer cheek vs. the
  main roof differ by more than 30°).
* `assign_max_residual` 0.20 m: equals the obstruction protrusion threshold, so
  anything a plane cannot explain is an obstruction candidate.
* `max_seam_offset` 0.6 m: about twice the DSM edge-smoothing width.
* Setback and clearance 0.30 m: common installer practice. Shading cut-off 30 %:
  beyond this a module rarely pays back.
