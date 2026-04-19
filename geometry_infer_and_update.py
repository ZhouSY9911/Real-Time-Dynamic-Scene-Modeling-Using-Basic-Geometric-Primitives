import trimesh
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


# ============================================================
# Data structure (Phase-1 geometry)
# ============================================================

@dataclass
class UpdatedGeometry:
    """
    Geometry hypothesis after update (Phase-1).

    shape:
        - "box"
        - "sphere"
        - "cylinder"

    confidence:
        - "low" | "medium" | "high"
    """
    shape: str
    confidence: str
    allow_upgrade: bool
    allow_downgrade: bool
    params: Dict[str, float]


# ============================================================
# Core API
# ============================================================

def infer_and_update_geometry(
    pts_fused: np.ndarray,
    prev_geom,
    min_points: int = 200,
    # --- PCA / shape thresholds
    axis_ratio_thresh: float = 2.5,
    radius_cv_thresh: float = 0.15,
    slenderness_thresh: float = 2.0,
) -> UpdatedGeometry:
    """
    Phase-1 geometry inference & update.

    Strategy:
      - Only attempt upgrade if allow_upgrade == True
      - Only attempt: box / sphere → cylinder
      - Cylinder upgrade is monotonic (no downgrade)

    Args:
        pts_fused: fused point cloud in object frame (N,3)
        prev_geom: InitGeometry or UpdatedGeometry
    """

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------
    if pts_fused.shape[0] < min_points:
        return _keep_previous(prev_geom)

    if not prev_geom.allow_upgrade:
        return _keep_previous(prev_geom)

    if prev_geom.shape == "cylinder":
        return _keep_previous(prev_geom)

    
    # --------------------------------------------------------
    # Try sphere / ellipsoid upgrade first
    # --------------------------------------------------------
    success_sph, sph_params, sph_score = _try_fit_sphere(pts_fused)
    if success_sph and prev_geom.shape == "box":
        return UpdatedGeometry(
            shape="sphere",
            confidence=_score_to_confidence(sph_score),
            allow_upgrade=True,
            allow_downgrade=True,
            params=sph_params,
        )


    # --------------------------------------------------------
    # Only try cylinder if NOT sphere-like
    # --------------------------------------------------------
    if not success_sph:
        success_cyl, cyl_params, cyl_score = _try_fit_cylinder(
            pts_fused,
            axis_ratio_thresh,
            radius_cv_thresh,
            slenderness_thresh,
        )

        if success_cyl:
            return UpdatedGeometry(
            shape="cylinder",
            confidence=_score_to_confidence(cyl_score),
            allow_upgrade=False,
            allow_downgrade=False,
            params=cyl_params,
        )

    return _keep_previous(prev_geom)

    


# ============================================================
# Cylinder inference
# ============================================================

def _try_fit_cylinder(
    pts: np.ndarray,
    axis_ratio_thresh: float,
    radius_cv_thresh: float,
    slenderness_thresh: float,
) -> Tuple[bool, Optional[Dict], float]:
    """
    Decide whether the point cloud supports a cylinder hypothesis.
    """

    axis, center, eigvals = _estimate_axis_pca(pts)
    radius, radius_std, height = _estimate_radius_height(
        pts, axis, center
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    axis_ratio = eigvals[0] / max(eigvals[1], 1e-6)
    radius_cv = radius_std / max(radius, 1e-6)
    slenderness = height / max(radius, 1e-6)

    # --------------------------------------------------------
    # Hard decision
    # --------------------------------------------------------
    if (
        axis_ratio < axis_ratio_thresh or
        radius_cv > radius_cv_thresh or
        slenderness < slenderness_thresh
    ):
        return False, None, 0.0

    # --------------------------------------------------------
    # Confidence score (0~1)
    # --------------------------------------------------------
    score = (
        0.4 * min(axis_ratio / 5.0, 1.0) +
        0.4 * (1.0 - radius_cv) +
        0.2 * min(slenderness / 5.0, 1.0)
    )

    params = {
        "axis": axis.astype(np.float32),
        "center": center.astype(np.float32),
        "radius": float(radius),
        "height": float(height),
        # --- debug / analysis
        "axis_ratio": float(axis_ratio),
        "radius_cv": float(radius_cv),
        "slenderness": float(slenderness),
    }

    return True, params, score


# ============================================================
# Sphere inference
# ============================================================
def _try_fit_sphere(
    pts: np.ndarray,
    iso_ratio_thresh: float = 2.0,
    radius_cv_thresh: float = 0.25,
    axis_ratio_reject: float = 1.8,
) -> Tuple[bool, Optional[Dict], float]:
    """
    Decide whether the point cloud supports a sphere / ellipsoid hypothesis.

    NOTE:
      - This includes mildly anisotropic ellipsoids (e.g. lemon-like objects)
      - This function MUST be evaluated before cylinder fitting
    """

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------
    center = np.mean(pts, axis=0)
    X = pts - center

    cov = np.cov(X.T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]

    # --------------------------------------------------------
    # Isotropy test (reject strong elongation)
    # --------------------------------------------------------
    iso_ratio = eigvals[0] / max(eigvals[2], 1e-6)
    if iso_ratio > iso_ratio_thresh:
        return False, None, 0.0

    # --------------------------------------------------------
    # Reject obvious cylinders
    # --------------------------------------------------------
    axis_ratio = eigvals[0] / max(eigvals[1], 1e-6)
    if axis_ratio > axis_ratio_reject:
        return False, None, 0.0

    # --------------------------------------------------------
    # Radial consistency
    # --------------------------------------------------------
    radii = np.linalg.norm(X, axis=1)
    r_mean = radii.mean()
    r_std = radii.std()
    r_cv = r_std / max(r_mean, 1e-6)

    if r_cv > radius_cv_thresh:
        return False, None, 0.0

    # --------------------------------------------------------
    # Confidence score
    # --------------------------------------------------------
    score = (
        0.5 * (1.0 - min(iso_ratio / iso_ratio_thresh, 1.0)) +
        0.5 * (1.0 - min(r_cv / radius_cv_thresh, 1.0))
    )

    params = {
        "center": center.astype(np.float32),
        "diameter": float(2.0 * r_mean),
        # --- debug
        "iso_ratio": float(iso_ratio),
        "radius_cv": float(r_cv),
    }

    return True, params, score




# ============================================================
# Geometry utilities
# ============================================================

def _estimate_axis_pca(
    pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA-based dominant axis estimation.
    """
    center = np.mean(pts, axis=0)
    X = pts - center

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    axis = eigvecs[:, 0]
    axis /= np.linalg.norm(axis) + 1e-6

    return axis, center, eigvals


def _estimate_radius_height(
    pts: np.ndarray,
    axis: np.ndarray,
    center: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Estimate cylinder radius / height given axis.
    """
    v = pts - center
    proj = v @ axis

    h_min, h_max = proj.min(), proj.max()
    height = h_max - h_min

    radial = v - np.outer(proj, axis)
    radii = np.linalg.norm(radial, axis=1)

    return radii.mean(), radii.std(), height


# ============================================================
# Helpers
# ============================================================

def _keep_previous(prev_geom) -> UpdatedGeometry:
    """
    Keep previous geometry unchanged.
    """
    if isinstance(prev_geom, UpdatedGeometry):
        return prev_geom

    return UpdatedGeometry(
        shape=prev_geom.shape,
        confidence=prev_geom.confidence,
        allow_upgrade=prev_geom.allow_upgrade,
        allow_downgrade=prev_geom.allow_downgrade,
        params=prev_geom.params,
    )


def _score_to_confidence(score: float) -> str:
    if score > 0.75:
        return "high"
    elif score > 0.55:
        return "medium"
    else:
        return "low"


def build_updated_geometry_mesh(
    geom,
    inflate_ratio: float = 1.05,
    min_extent: float = 0.01,
    cylinder_sections: int = 32,
) -> trimesh.Trimesh:
    """
    Build geometry mesh from UpdatedGeometry.

    Supported shapes:
      - box
      - sphere
      - cylinder

    Notes:
      - Mesh is centered at origin (object frame)
      - Cylinder axis is +Z (canonical)
      - Orientation alignment should be handled outside
    """

    shape = geom.shape.lower()
    params = geom.params

    # =====================================================
    # BOX
    # =====================================================
    if shape == "box":
        sx = params["sx"] * inflate_ratio
        sy = params["sy"] * inflate_ratio
        sz = params["sz"] * inflate_ratio

        sx = max(sx, min_extent)
        sy = max(sy, min_extent)
        sz = max(sz, min_extent)

        mesh = trimesh.creation.box(
            extents=(sx, sy, sz)
        )
        return mesh

    # =====================================================
    # SPHERE
    # =====================================================
    elif shape == "sphere":
        diameter = params.get("diameter", 0.0) * inflate_ratio
        diameter = max(diameter, min_extent)

        radius = diameter * 0.5

        mesh = trimesh.creation.icosphere(
            radius=radius,
            subdivisions=3
        )
        return mesh

    # =====================================================
    # CYLINDER
    # =====================================================
    elif shape == "cylinder":
        radius = params["radius"] * inflate_ratio
        height = params["height"] * inflate_ratio

        radius = max(radius, min_extent)
        height = max(height, min_extent)

        mesh = trimesh.creation.cylinder(
            radius=radius,
            height=height,
            sections=cylinder_sections
        )

        # trimesh cylinder is centered at origin, axis = Z
        return mesh

    # =====================================================
    # Unsupported shape
    # =====================================================
    else:
        raise ValueError(
            f"Unsupported updated geometry shape: {geom.shape}"
        )
