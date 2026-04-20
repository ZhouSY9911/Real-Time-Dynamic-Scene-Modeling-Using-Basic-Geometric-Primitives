from __future__ import annotations

import numpy as np
import trimesh
import cv2
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any


# ============================================================
# Data structure
# ============================================================

@dataclass
class InitGeometry:
    """
    Initial geometry hypothesis (Phase-0).

    shape:
        - "box"
        - "sphere"
        - "cylinder"

    confidence:
        - currently low by design

    allow_upgrade:
        - MUST be True in Phase-0

    allow_downgrade:
        - True for sphere/cylinder priors
        - False for conservative box fallback
    """
    shape: str
    confidence: str
    allow_upgrade: bool
    allow_downgrade: bool
    params: Dict[str, Any]


# ============================================================
# Utilities
# ============================================================

def robust_box_from_points(
    pts_cam: np.ndarray,
    q_lo: float = 0.02,
    q_hi: float = 0.98,
    min_extent: float = 0.01,
) -> Dict[str, float]:
    """
    Conservative axis-aligned box from incomplete point cloud.
    """
    if pts_cam is None or pts_cam.shape[0] == 0:
        return {"sx": min_extent, "sy": min_extent, "sz": min_extent}

    pts = pts_cam[np.isfinite(pts_cam).all(axis=1)]
    if pts.shape[0] == 0:
        return {"sx": min_extent, "sy": min_extent, "sz": min_extent}

    center = np.median(pts, axis=0)
    pts0 = pts - center[None, :]

    lo = np.quantile(pts0, q_lo, axis=0)
    hi = np.quantile(pts0, q_hi, axis=0)

    extents = np.maximum(hi - lo, min_extent)

    return {
        "sx": float(extents[0]),
        "sy": float(extents[1]),
        "sz": float(extents[2]),
    }


def safe_contour_analysis(mask: np.ndarray) -> Tuple[float, float, int, bool]:
    """
    Return:
        circularity, aspect_ratio, corner_count, valid
    """
    if mask is None:
        return 0.0, 0.0, 0, False

    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0, 0.0, 0, False

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    if area <= 1.0 or perimeter <= 1e-6:
        return 0.0, 0.0, 0, False

    circularity = float(4.0 * np.pi * area / (perimeter ** 2))

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corner_count = int(len(approx))

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(max(w, h) / max(1.0, min(w, h)))

    return circularity, aspect_ratio, corner_count, True


def _estimate_3d_extents(pts_cam: np.ndarray, min_extent: float = 0.01) -> Tuple[float, float, float]:
    box = robust_box_from_points(pts_cam, min_extent=min_extent)
    return box["sx"], box["sy"], box["sz"]


def _rasterize_2d(points_2d: np.ndarray, res: float = 0.003) -> Optional[np.ndarray]:
    if points_2d is None or points_2d.shape[0] < 10:
        return None

    pts = points_2d[np.isfinite(points_2d).all(axis=1)]
    if pts.shape[0] < 10:
        return None

    mn = pts.min(axis=0)
    ptsi = ((pts - mn[None, :]) / max(res, 1e-6)).astype(np.int32)

    w = int(ptsi[:, 0].max()) + 3
    h = int(ptsi[:, 1].max()) + 3

    if w <= 2 or h <= 2 or w > 2048 or h > 2048:
        return None

    img = np.zeros((h, w), dtype=np.uint8)
    img[ptsi[:, 1], ptsi[:, 0]] = 255

    kernel = np.ones((3, 3), dtype=np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return img


def _projection_features(points_2d: np.ndarray) -> Dict[str, float]:
    """
    Features from a 2D point set / occupancy projection.
    """
    out = {
        "pca_ratio": 999.0,
        "circularity": 0.0,
        "rectangularity": 0.0,
        "line_support": 0.0,
        "valid": 0.0,
    }

    if points_2d is None or points_2d.shape[0] < 10:
        return out

    pts = points_2d[np.isfinite(points_2d).all(axis=1)]
    if pts.shape[0] < 10:
        return out

    # PCA ratio
    center = np.mean(pts, axis=0)
    X = pts - center[None, :]
    cov = np.cov(X.T)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]
    pca_ratio = float(eigvals[0] / max(eigvals[1], 1e-6))

    img = _rasterize_2d(pts)
    if img is None:
        out["pca_ratio"] = pca_ratio
        return out

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        out["pca_ratio"] = pca_ratio
        return out

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    if area <= 1.0 or perimeter <= 1e-6:
        out["pca_ratio"] = pca_ratio
        return out

    circularity = float(4.0 * np.pi * area / (perimeter ** 2))

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = float(max(w * h, 1.0))
    rectangularity = float(area / bbox_area)

    lines = cv2.HoughLinesP(
        img,
        rho=1,
        theta=np.pi / 180.0,
        threshold=20,
        minLineLength=max(8, int(0.15 * max(w, h))),
        maxLineGap=3,
    )
    line_support = 0.0
    if lines is not None:
        total_len = 0.0
        for l in lines[:, 0]:
            x1, y1, x2, y2 = l.tolist()
            total_len += float(np.hypot(x2 - x1, y2 - y1))
        line_support = float(min(total_len / max(perimeter, 1.0), 1.0))

    out.update({
        "pca_ratio": pca_ratio,
        "circularity": circularity,
        "rectangularity": rectangularity,
        "line_support": line_support,
        "valid": 1.0,
    })
    return out


def _xyz_projection_features(pts_cam: np.ndarray) -> Dict[str, Dict[str, float]]:
    if pts_cam is None or pts_cam.shape[0] < 10:
        return {
            "xy": _projection_features(np.zeros((0, 2), dtype=np.float32)),
            "xz": _projection_features(np.zeros((0, 2), dtype=np.float32)),
            "yz": _projection_features(np.zeros((0, 2), dtype=np.float32)),
        }

    return {
        "xy": _projection_features(pts_cam[:, [0, 1]]),
        "xz": _projection_features(pts_cam[:, [0, 2]]),
        "yz": _projection_features(pts_cam[:, [1, 2]]),
    }


# ============================================================
# Core API
# ============================================================

def init_geometry_from_first_frame(
    pts_cam: np.ndarray,
    mask: np.ndarray,
    min_points: int = 50,

    # 2D contour prior
    circularity_thresh: float = 0.82,
    aspect_ratio_thresh: float = 1.25,

    # 3D aspect priors
    round_ratio_thresh_xy: float = 1.30,
    round_ratio_thresh_xz: float = 1.80,
    round_ratio_thresh_yz: float = 1.80,

    # cylinder prior
    cylinder_cross_ratio_thresh: float = 1.30,
    cylinder_height_ratio_thresh: float = 1.20,

    min_extent: float = 0.01,
) -> InitGeometry:
    """
    Conservative Phase-0 initialization.

    Logic:
      - Always compute a fallback box
      - If first-frame evidence strongly suggests round object -> low-confidence sphere
      - If first-frame evidence strongly suggests cylinder -> low-confidence cylinder
      - Otherwise fallback to low-confidence box
    """
    box_params = robust_box_from_points(
        pts_cam=pts_cam,
        q_lo=0.02,
        q_hi=0.98,
        min_extent=min_extent,
    )

    sx = float(box_params["sx"])
    sy = float(box_params["sy"])
    sz = float(box_params["sz"])

    if pts_cam is None or pts_cam.shape[0] < min_points:
        return InitGeometry(
            shape="box",
            confidence="low",
            allow_upgrade=True,
            allow_downgrade=False,
            params=box_params,
        )

    circularity, aspect_ratio, corner_count, contour_valid = safe_contour_analysis(mask)
    proj = _xyz_projection_features(pts_cam)

    r_xy = max(sx, sy) / max(min(sx, sy), 1e-6)
    r_xz = max(sx, sz) / max(min(sx, sz), 1e-6)
    r_yz = max(sy, sz) / max(min(sy, sz), 1e-6)

    # --------------------------------------------------------
    # Sphere-like prior
    # --------------------------------------------------------
    is_round_2d = (
        contour_valid and
        circularity > circularity_thresh and
        aspect_ratio < aspect_ratio_thresh
    )

    is_round_3d = (
        r_xy < round_ratio_thresh_xy and
        r_xz < round_ratio_thresh_xz and
        r_yz < round_ratio_thresh_yz
    )

    num_circle_like_proj = 0
    for k in ["xy", "xz", "yz"]:
        f = proj[k]
        if f["valid"] > 0.5 and f["pca_ratio"] < 1.30 and f["circularity"] > 0.72:
            num_circle_like_proj += 1

    is_sphere_like = is_round_3d and (is_round_2d or num_circle_like_proj >= 2)

    # --------------------------------------------------------
    # Cylinder-like prior
    # Assume common tabletop case: cylinder axis often along Z.
    # Keep this conservative.
    # --------------------------------------------------------
    cross_ratio_xy = max(sx, sy) / max(min(sx, sy), 1e-6)
    radial_xy = 0.5 * (sx + sy)
    height_to_radial = sz / max(radial_xy, 1e-6)

    xy_circle_like = proj["xy"]["valid"] > 0.5 and proj["xy"]["pca_ratio"] < 1.30 and proj["xy"]["circularity"] > 0.72
    xz_rect_like = proj["xz"]["valid"] > 0.5 and proj["xz"]["pca_ratio"] > 1.30 and proj["xz"]["rectangularity"] > 0.55
    yz_rect_like = proj["yz"]["valid"] > 0.5 and proj["yz"]["pca_ratio"] > 1.30 and proj["yz"]["rectangularity"] > 0.55

    is_cylinder_like_3d = (
        cross_ratio_xy < cylinder_cross_ratio_thresh and
        height_to_radial > cylinder_height_ratio_thresh
    )

    is_cylinder_like = is_cylinder_like_3d and (xy_circle_like or (xz_rect_like and yz_rect_like))

    # --------------------------------------------------------
    # Conservative decision
    # --------------------------------------------------------
    if is_cylinder_like and not is_sphere_like:
        radius = max(0.25 * (sx + sy), 0.5 * min_extent)
        return InitGeometry(
            shape="cylinder",
            confidence="low",
            allow_upgrade=True,
            allow_downgrade=True,
            params={
                "radius": float(radius),
                "height": float(max(sz, min_extent)),
                "axis": np.array([0.0, 0.0, 1.0], dtype=np.float32),
                "sx": float(sx),
                "sy": float(sy),
                "sz": float(sz),
                "circularity": float(circularity),
                "aspect_ratio": float(aspect_ratio),
                "corner_count": int(corner_count),
                "r_xy": float(r_xy),
                "r_xz": float(r_xz),
                "r_yz": float(r_yz),
            },
        )

    if is_sphere_like:
        diameter = float(max(np.median([sx, sy, sz]), min_extent))
        return InitGeometry(
            shape="sphere",
            confidence="low",
            allow_upgrade=True,
            allow_downgrade=True,
            params={
                "diameter": diameter,
                "sx": float(sx),
                "sy": float(sy),
                "sz": float(sz),
                "circularity": float(circularity),
                "aspect_ratio": float(aspect_ratio),
                "corner_count": int(corner_count),
                "r_xy": float(r_xy),
                "r_xz": float(r_xz),
                "r_yz": float(r_yz),
            },
        )

    return InitGeometry(
        shape="box",
        confidence="low",
        allow_upgrade=True,
        allow_downgrade=False,
        params=box_params,
    )


# ============================================================
# Build initial mesh
# ============================================================

def build_init_geometry_mesh(
    init_geom: InitGeometry,
    inflate_ratio: float = 1.10,
    min_extent: float = 0.01,
    cylinder_sections: int = 32,
) -> trimesh.Trimesh:
    """
    Build conservative init geometry mesh.

    Notes:
      - Mesh is centered at origin (object frame)
      - Cylinder canonical axis is +Z
      - Init geometry is intentionally conservative / inflated
    """
    shape = init_geom.shape.lower()
    params = init_geom.params

    if shape == "box":
        sx = max(params["sx"] * inflate_ratio, min_extent)
        sy = max(params["sy"] * inflate_ratio, min_extent)
        sz = max(params["sz"] * inflate_ratio, min_extent)

        return trimesh.creation.box(extents=(sx, sy, sz))

    elif shape == "sphere":
        diameter = max(params["diameter"] * inflate_ratio, min_extent)
        radius = 0.5 * diameter
        return trimesh.creation.icosphere(radius=radius, subdivisions=3)

    elif shape == "cylinder":
        radius = max(params["radius"] * inflate_ratio, 0.5 * min_extent)
        height = max(params["height"] * inflate_ratio, min_extent)
        return trimesh.creation.cylinder(
            radius=radius,
            height=height,
            sections=cylinder_sections,
        )

    else:
        raise ValueError(f"Unsupported init geometry shape: {init_geom.shape}")