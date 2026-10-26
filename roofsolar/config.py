"""Central configuration for the roof reconstruction / PV layout pipeline.

All lengths are metres, all angles are degrees unless stated otherwise.
Every threshold used anywhere in the pipeline lives here so it can be
inspected, justified and tuned in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class PreprocessConfig:
    # Robust spike / pit detection: |z - median_k(z)| > max(k_mad * MAD, min_abs)
    outlier_window: int = 5
    outlier_k_mad: float = 6.0
    outlier_min_abs: float = 0.5
    # Edge-preserving smoothing applied before normal estimation
    median_size: int = 3


@dataclass
class BuildingConfig:
    dtm_radius: float = 12.0          # morphological opening radius for ground (> half building width)
    dtm_downsample: int = 4           # opening done on a decimated grid for speed
    min_height: float = 2.5           # nDSM threshold for "elevated object"
    exg_threshold: float | None = None  # None -> Otsu on elevated pixels
    min_building_area: float = 40.0   # m^2
    opening_radius: float = 0.3       # remove thin vegetation spurs


@dataclass
class SegmentationConfig:
    normal_sigma: float = 1.5         # px, gaussian used for normals (on masked DSM)
    planarity_window: int = 5         # px, local plane fit window for seed quality
    seed_max_residual: float = 0.03   # m, local RMS for a pixel to be a seed
    grow_angle: float = 12.0          # deg, normal deviation allowed while growing
    grow_dist: float = 0.12           # m, point-to-plane distance allowed while growing
    min_region_area: float = 0.6      # m^2, smaller regions are dissolved
    merge_angle: float = 6.0          # deg
    merge_dist: float = 0.10          # m (RMS of mutual residual)
    assign_max_residual: float = 0.20 # m, pixels further than this from every plane stay unassigned
    step_edge_height: float = 0.35    # m, plane height jump that marks a vertical step (no shared ridge)
    envelope_band: float = 0.8        # m, band around boundaries re-labelled by the envelope rule
    simplify_tolerance: float = 0.20  # m, coverage-preserving polygon simplification
    max_seam_offset: float = 0.6      # m, max distance boundary <-> plane-intersection line for a true seam
    flat_slope: float = 8.0           # deg, below this a face is "flat"


@dataclass
class ObstructionConfig:
    min_protrusion: float = 0.20      # m above fitted plane
    min_area: float = 0.04            # m^2
    rgb_k_mad: float = 5.0            # colour anomaly threshold (robust z-score in Lab)
    rgb_min_delta_e: float = 6.0      # CIE76 floor
    chroma_k_mad: float = 3.0         # chromatic (a*b*) anomaly threshold, robust sigmas
    chroma_min: float = 6.0           # minimum a*b* shift
    glass_delta_a: float = -6.0       # a* below roof median by this much -> glazing
    glazed_face_fraction: float = 0.4 # face mostly glazing -> whole face is a skylight/lantern
    rgb_min_area: float = 0.12        # m^2
    rgb_max_area: float = 12.0        # m^2
    small_face_area: float = 1.5      # m^2: tiny raised faces (chimney tops, lantern facets)
    edge_margin: float = 0.25         # m: anomalies touching face edges are often shadows/edge effects


@dataclass
class SolarConfig:
    panel_length: float = 1.722       # m (long side) - typical 400-430 Wp module
    panel_width: float = 1.134        # m (short side)
    panel_power_wp: float = 425.0
    panel_gap: float = 0.02           # m clamp gap between modules
    edge_setback: float = 0.30        # m from every roof-face edge (eave, ridge, hip, verge)
    obstruction_clearance: float = 0.30  # m buffer around every obstruction
    max_slope: float = 60.0
    min_relative_yield: float = 0.75  # vs. best possible orientation at this latitude
    diffuse_fraction: float = 0.5     # annual diffuse share of GHI (UK ~0.5)
    offset_step: float = 0.05         # m search step for grid phase
    min_panels_per_face: int = 1
    max_shading_loss: float = 0.30   # drop modules losing more than this share of annual beam energy
    latitude: float | None = None     # None -> derived from raster CRS
    longitude: float | None = None


@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    building: BuildingConfig = field(default_factory=BuildingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    obstruction: ObstructionConfig = field(default_factory=ObstructionConfig)
    solar: SolarConfig = field(default_factory=SolarConfig)

    def to_dict(self) -> dict:
        return asdict(self)
