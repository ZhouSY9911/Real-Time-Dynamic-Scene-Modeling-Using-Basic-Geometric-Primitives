import numpy as np
import trimesh
import cv2
from dataclasses import dataclass
from typing import Dict, Tuple


# ===============================
# Data structure
# ===============================

@dataclass
class InitGeometry:
    """
    Initial geometry hypothesis (Phase-0).

    NOTE:
      - confidence is always "low"
      - shape is always "box"
      - allow_upgrade MUST be True
    """
    shape: str                  # always "box"
    confidence: str             # always "low"
    allow_upgrade: bool
    allow_downgrade: bool
    params: Dict[str, float]    # rough box extents (sx, sy, sz)


# ===============================
# Utilities
# ===============================

def robust_box_from_points(
    pts_cam: np.ndarray,
    q_lo: float = 0.02,
    q_hi: float = 0.98,
    min_extent: float = 0.01,
) -> Dict[str, float]:
    """
    Compute a conservative axis-aligned box from incomplete point cloud.

    Returns:
        dict with keys: sx, sy, sz
    """
    center = np.median(pts_cam, axis=0)
    pts = pts_cam - center

    lo = np.quantile(pts, q_lo, axis=0)
    hi = np.quantile(pts, q_hi, axis=0)

    extents = hi - lo
    extents = np.maximum(extents, min_extent)

    return {
        "sx": float(extents[0]),
        "sy": float(extents[1]),
        "sz": float(extents[2]),
    }


# ===============================
# Core API
# ===============================

def init_geometry_from_first_frame(
    pts_cam: np.ndarray,
    mask: np.ndarray,
    min_points: int = 50,
    # --- 2D thresholds
    circularity_thresh: float = 0.85,
    aspect_ratio_thresh: float = 1.2,
    min_corner_count: int = 8,
    # --- 3D thresholds
    xy_ratio_thresh: float = 1.3,
) -> InitGeometry:
    """
    Phase-0 geometry initialization (FINAL).

    Strategy:
      - Use 2D mask contour to detect circle-like footprint
      - Use 3D point cloud to verify isotropic XY scale
      - ONLY if both agree → circle-like init
      - Otherwise → fallback to box

    Notes:
      - This stage does NOT decide cylinder vs box
      - Geometry upgrade is deferred to later phases
    """

    # =========================================================
    # 0. Safety check
    # =========================================================
    if pts_cam.shape[0] < min_points:
        raise RuntimeError(
            f"Too few points for geometry init: {pts_cam.shape[0]}"
        )

    # =========================================================
    # 1. Robust 3D box estimation (ALWAYS needed)
    # =========================================================
    box_params = robust_box_from_points(pts_cam)
    sx = box_params["sx"]
    sy = box_params["sy"]
    sz = box_params["sz"]

    # =========================================================
    # 2. 3D circle-like test (XY isotropy)
    # =========================================================
    s_max = max(sx, sy)
    s_min = max(min(sx, sy), 1e-6)
    is_circle_3d = (s_max / s_min) < xy_ratio_thresh

    # =========================================================
    # 3. 2D mask contour analysis
    # =========================================================
    is_circle_2d = False
    circularity = 0.0
    aspect_ratio = 0.0
    corner_count = 0

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter > 1e-6 and area > 1.0:
            circularity = 4.0 * np.pi * area / (perimeter ** 2)

            epsilon = 0.02 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            corner_count = len(approx)

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / max(1.0, min(w, h))

            is_circle_2d = (
                circularity > circularity_thresh and
                aspect_ratio < aspect_ratio_thresh and
                corner_count >= min_corner_count
            )

    # =========================================================
    # 4. Final decision (2D + 3D agreement)
    # =========================================================
    if is_circle_2d and is_circle_3d:
        # Conservative diameter from 3D
        diameter = s_max

        return InitGeometry(
            shape="sphere",              # footprint-level shape
            confidence="low",
            allow_upgrade=True,
            allow_downgrade=True,
            params={
                "diameter": float(diameter),
                # --- debug info (optional but useful)
                "circularity": float(circularity),
                "aspect_ratio": float(aspect_ratio),
                "corner_count": int(corner_count),
            },
        )

    # =========================================================
    # 5. Fallback: conservative box
    # =========================================================
    return InitGeometry(
        shape="box",
        confidence="low",
        allow_upgrade=True,
        allow_downgrade=False,
        params=box_params,
    )


# ===============================
# build initial mesh
# ===============================

def build_init_geometry_mesh(
    init_geom: InitGeometry,
    inflate_ratio: float = 1.1,
    min_extent: float = 0.01,
) -> trimesh.Trimesh:
    """
    Build conservative init geometry mesh (box / sphere).

    - Mesh is centered at origin (object frame)
    - Geometry is intentionally inflated for safety
    """

    shape = init_geom.shape.lower()
    params = init_geom.params

    # =====================================================
    # BOX
    # =====================================================
    if shape == "box":
        sx = params["sx"] 
        sy = params["sy"] 
        sz = params["sz"] 

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
        diameter = params["diameter"] * inflate_ratio
        diameter = max(diameter, min_extent)

        radius = diameter * 0.5

        mesh = trimesh.creation.icosphere(
            radius=radius,
            subdivisions=3
        )
        return mesh

    # =====================================================
    # Unsupported shape
    # =====================================================
    else:
        raise ValueError(
            f"Unsupported init geometry shape: {init_geom.shape}"
        )
