from __future__ import annotations

import os
import numpy as np
import trimesh
import cv2
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any, List


# ============================================================
# Data structure
# ============================================================

@dataclass
class UpdatedGeometry:
    """
    Geometry hypothesis after update.

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
    params: Dict[str, Any]


@dataclass
class _ShapeCandidate:
    shape: str
    score: float
    confidence: str
    params: Dict[str, Any]


# ============================================================
# Core API
# ============================================================

def infer_and_update_geometry(
    pts_fused: np.ndarray,
    prev_geom,
    plane_normal: Optional[np.ndarray] = None,   # 保留接口兼容
    min_points: int = 180,
    radius_profile_cv_thresh: float = 0.12,
    radius_range_ratio_thresh: float = 0.22,
    shape_score_margin: float = 0.06,
    debug_save_dir: Optional[str] = None,
    debug_prefix: str = "geom",
) -> UpdatedGeometry:
    """
    Projection-fitting-first inference.

    Main logic:
      1) preprocess fused point cloud
      2) transform to PCA frame
      3) fit circle / rectangle on 3 PCA projections
      4) infer prior shape:
           - >=2 circle-like projections => sphere prior
           - ==1 circle-like projection => cylinder prior
           - 0 circle-like projections => box prior
      5) if cylinder prior:
           use the normal direction of the circular projection as axis candidate,
           then analyze radius variation along axis:
             - nearly constant radius => cylinder
             - obvious radius change => sphere
      6) combine prior with lightweight 3D fitting candidates
    """
    if pts_fused is None or pts_fused.shape[0] < min_points:
        return _keep_previous(prev_geom)

    if not getattr(prev_geom, "allow_upgrade", True):
        return _keep_previous(prev_geom)

    pts = np.asarray(pts_fused, dtype=np.float64)
    pts = _remove_invalid_points(pts)
    if pts.shape[0] < min_points:
        return _keep_previous(prev_geom)

    pts = _robust_trim_points(pts, keep_ratio=0.97)
    if pts.shape[0] < min_points:
        return _keep_previous(prev_geom)

    pts = _voxel_downsample_numpy(pts, voxel_size=_estimate_voxel_size(pts))
    if pts.shape[0] < max(120, min_points // 2):
        return _keep_previous(prev_geom)

    # --------------------------------------------------------
    # PCA frame
    # --------------------------------------------------------
    pts_pca, R_pca, center_world, pca_feats = _transform_points_to_pca_frame(pts)

    p12 = pts_pca[:, [0, 1]]
    p13 = pts_pca[:, [0, 2]]
    p23 = pts_pca[:, [1, 2]]

    projection_points_2d = {
        "p12": p12,
        "p13": p13,
        "p23": p23,
    }

    projection_info = _analyze_three_pca_projections(pts_pca)

    # debug projection images
    if debug_save_dir is not None:
        try:
            save_projection_bundle_debug_with_text(
                projection_points_2d=projection_points_2d,
                projection_info=projection_info,
                save_dir=debug_save_dir,
                prefix=debug_prefix,
                img_size=512,
            )
        except Exception:
            pass

    # --------------------------------------------------------
    # Prior from projection fitting
    # --------------------------------------------------------
    prior_info = infer_shape_by_projection_fitting(
        pts_pca=pts_pca,
        radius_cv_thresh=radius_profile_cv_thresh,
        radius_range_ratio_thresh=radius_range_ratio_thresh,
    )

    prior_shape = prior_info["shape"]
    prev_shape = getattr(prev_geom, "shape", "box")

    # --------------------------------------------------------
    # Candidate fitting in PCA frame
    # --------------------------------------------------------
    cand_sphere = _make_sphere_candidate(
        pts=pts_pca,
        projection_info=projection_info,
        pca_feats=pca_feats,
    )
    cand_cylinder = _make_cylinder_candidate(
        pts=pts_pca,
        projection_info=projection_info,
        pca_feats=pca_feats,
        prior_info=prior_info,
        radius_profile_cv_thresh=radius_profile_cv_thresh,
        radius_range_ratio_thresh=radius_range_ratio_thresh,
    )
    cand_box = _make_box_candidate(
        pts=pts_pca,
        projection_info=projection_info,
        pca_feats=pca_feats,
    )

    # --------------------------------------------------------
    # Apply projection prior to candidate scores
    # --------------------------------------------------------
    cand_sphere = _apply_prior_bias(cand_sphere, prior_shape)
    cand_cylinder = _apply_prior_bias(cand_cylinder, prior_shape)
    cand_box = _apply_prior_bias(cand_box, prior_shape)

    # --------------------------------------------------------
    # Select best
    # --------------------------------------------------------
    if prior_shape == "sphere":
        best = _choose_best_with_prior(
            candidates=[cand_sphere, cand_cylinder, cand_box],
            prev_shape=prev_shape,
            score_margin=shape_score_margin,
        )
    elif prior_shape == "cylinder":
        best = _choose_best_with_prior(
            candidates=[cand_cylinder, cand_sphere, cand_box],
            prev_shape=prev_shape,
            score_margin=shape_score_margin,
        )
    else:
        best = _choose_best_with_prior(
            candidates=[cand_box, cand_cylinder, cand_sphere],
            prev_shape=prev_shape,
            score_margin=shape_score_margin,
        )

    # Protect stable cylinder
    if prev_shape == "cylinder":
        if (best.shape != "cylinder") and ((best.score - cand_cylinder.score) < 0.08):
            best = cand_cylinder

    final_params = _convert_candidate_params_back_to_world(
        cand=best,
        R_pca=R_pca,
        center_world=center_world,
    )

    # Attach debug/meta info
    final_params["projection_prior_shape"] = prior_shape
    final_params["projection_prior_confidence"] = prior_info.get("confidence", "low")
    final_params["projection_circle_views"] = prior_info.get("circle_views", [])
    final_params["projection_info"] = projection_info
    final_params["axis_analysis"] = prior_info.get("axis_analysis", None)

    return UpdatedGeometry(
        shape=best.shape,
        confidence=best.confidence,
        allow_upgrade=(best.shape != "cylinder"),
        allow_downgrade=(best.shape != "box"),
        params=final_params,
    )


# ============================================================
# Candidate builders
# ============================================================

def _make_sphere_candidate(
    pts: np.ndarray,
    projection_info: Dict[str, Dict[str, float]],
    pca_feats: Dict[str, float],
) -> _ShapeCandidate:
    center, radius, residual_std = _fit_sphere_least_squares(pts)

    if center is None or radius is None or radius <= 1e-8:
        return _ShapeCandidate("sphere", 0.0, "low", {})

    d = np.linalg.norm(pts - center[None, :], axis=1)
    residual = np.abs(d - radius)

    obj_scale = _object_scale(pts)
    residual_rmse = float(np.sqrt(np.mean(residual ** 2)))
    residual_nrmse = float(residual_rmse / max(obj_scale, 1e-6))

    circle_views = []
    circle_scores = []
    rect_scores = []

    for k in ["p12", "p13", "p23"]:
        info = projection_info[k]
        if info["shape_label"] == "circle":
            circle_views.append(k)
        circle_scores.append(float(info["circle_score"]))
        rect_scores.append(float(info["rect_score"]))

    circle_count = len(circle_views)
    score_circle_views = float(np.mean(circle_scores)) if len(circle_scores) > 0 else 0.0
    score_rect_penalty = float(np.mean(rect_scores)) if len(rect_scores) > 0 else 0.0

    score_fit = float(np.clip(1.0 - residual_nrmse / 0.08, 0.0, 1.0))
    score_iso = float(np.clip(1.0 - (pca_feats["iso_ratio"] - 1.0) / 2.5, 0.0, 1.0))
    score_axis = float(np.clip(1.0 - (pca_feats["axis_ratio"] - 1.0) / 1.5, 0.0, 1.0))
    score_circle_count = float(np.clip(circle_count / 2.0, 0.0, 1.0))

    score = (
        0.45 * score_fit +
        0.18 * score_iso +
        0.12 * score_axis +
        0.17 * score_circle_count +
        0.12 * score_circle_views
    ) - 0.08 * max(score_rect_penalty - 0.65, 0.0)
    score = float(np.clip(score, 0.0, 1.0))

    params = {
        "center": center.astype(np.float32),
        "diameter": float(2.0 * radius),
        "fit_score": float(score_fit),
        "residual_rmse": float(residual_rmse),
        "residual_nrmse": float(residual_nrmse),
        "circle_count": int(circle_count),
        "circle_views": list(circle_views),
        "circle_views_score": float(score_circle_views),
        "iso_ratio": float(pca_feats["iso_ratio"]),
        "axis_ratio": float(pca_feats["axis_ratio"]),
        "mid_ratio": float(pca_feats["mid_ratio"]),
        "score_total": float(score),
    }

    return _ShapeCandidate(
        shape="sphere",
        score=score,
        confidence=_score_to_confidence(score),
        params=params,
    )


def _make_cylinder_candidate(
    pts: np.ndarray,
    projection_info: Dict[str, Dict[str, float]],
    pca_feats: Dict[str, float],
    prior_info: Dict[str, Any],
    radius_profile_cv_thresh: float,
    radius_range_ratio_thresh: float,
) -> _ShapeCandidate:
    # axis from projection prior if available, otherwise fallback to PCA x-axis
    axis_name = None
    axis = None

    axis_analysis = prior_info.get("axis_analysis", None)
    if axis_analysis is not None:
        axis_name = axis_analysis.get("axis_name", None)
        axis = np.asarray(axis_analysis.get("axis_vector", None), dtype=np.float64) \
            if axis_analysis.get("axis_vector", None) is not None else None

    if axis is None or axis.shape != (3,):
        axis_name = "x"
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    axis = axis / (np.linalg.norm(axis) + 1e-12)
    center = np.mean(pts, axis=0)

    v = pts - center[None, :]
    proj = v @ axis
    radial = v - np.outer(proj, axis)
    radii = np.linalg.norm(radial, axis=1)

    if radii.shape[0] < 20:
        return _ShapeCandidate("cylinder", 0.0, "low", {})

    radius = float(np.median(radii))
    if radius <= 1e-8:
        return _ShapeCandidate("cylinder", 0.0, "low", {})

    proj_q05, proj_q95 = np.quantile(proj, [0.05, 0.95])
    height = float(proj_q95 - proj_q05)

    residual = np.abs(radii - radius)
    obj_scale = _object_scale(pts)
    residual_rmse = float(np.sqrt(np.mean(residual ** 2)))
    residual_nrmse = float(residual_rmse / max(obj_scale, 1e-6))

    axis_ratio = float(pca_feats["axis_ratio"])
    height_to_diameter = float(height / max(2.0 * radius, 1e-6))

    profile = _compute_radius_profile_along_axis(
        pts=pts,
        axis=axis,
        num_slices=9,
        min_points_per_slice=20,
    )

    if profile["valid"] < 0.5:
        slice_radius_cv = 1.0
        radius_range_ratio = 1.0
    else:
        slice_radius_cv = float(profile["radius_cv"])
        radius_range_ratio = float(profile["radius_range_ratio"])

    circle_views = prior_info.get("circle_views", [])
    circle_count = len(circle_views)

    rect_like_count = 0
    rect_scores = []
    circle_scores = []

    for k in ["p12", "p13", "p23"]:
        info = projection_info[k]
        if info["shape_label"] == "rectangle":
            rect_like_count += 1
        rect_scores.append(float(info["rect_score"]))
        circle_scores.append(float(info["circle_score"]))

    rect_score = float(np.mean(rect_scores)) if len(rect_scores) > 0 else 0.0
    circle_score = float(np.mean(circle_scores)) if len(circle_scores) > 0 else 0.0

    score_fit = float(np.clip(1.0 - residual_nrmse / 0.08, 0.0, 1.0))
    score_axis = float(np.clip((axis_ratio - 1.0) / 2.5, 0.0, 1.0))
    score_slice = float(np.clip(1.0 - slice_radius_cv / max(radius_profile_cv_thresh * 2.0, 1e-6), 0.0, 1.0))
    score_range = float(np.clip(1.0 - radius_range_ratio / max(radius_range_ratio_thresh * 2.0, 1e-6), 0.0, 1.0))
    score_h2d = float(np.clip((height_to_diameter - 0.5) / 1.7, 0.0, 1.0))
    score_proj_pattern = float(np.clip(
        0.50 * min(circle_count, 1) + 0.50 * min(rect_like_count / 2.0, 1.0),
        0.0, 1.0
    ))

    score = (
        0.24 * score_fit +
        0.16 * score_axis +
        0.20 * score_slice +
        0.12 * score_range +
        0.10 * score_h2d +
        0.12 * score_proj_pattern +
        0.03 * rect_score +
        0.03 * circle_score
    )

    # penalty if radius varies too much => more sphere-like
    if slice_radius_cv > radius_profile_cv_thresh:
        score -= 0.10 * np.clip((slice_radius_cv - radius_profile_cv_thresh) / 0.20, 0.0, 1.0)
    if radius_range_ratio > radius_range_ratio_thresh:
        score -= 0.10 * np.clip((radius_range_ratio - radius_range_ratio_thresh) / 0.30, 0.0, 1.0)

    score = float(np.clip(score, 0.0, 1.0))

    params = {
        "axis": axis.astype(np.float32),
        "axis_name_pca": axis_name,
        "center": center.astype(np.float32),
        "radius": float(radius),
        "height": float(height),
        "fit_score": float(score_fit),
        "residual_rmse": float(residual_rmse),
        "residual_nrmse": float(residual_nrmse),
        "axis_ratio": float(axis_ratio),
        "slice_radius_cv": float(slice_radius_cv),
        "radius_range_ratio": float(radius_range_ratio),
        "height_to_diameter": float(height_to_diameter),
        "circle_count": int(circle_count),
        "circle_views": list(circle_views),
        "rect_like_count": int(rect_like_count),
        "rect_score": float(rect_score),
        "circle_score": float(circle_score),
        "radius_profile_cv": float(slice_radius_cv),
        "score_total": float(score),
    }

    return _ShapeCandidate(
        shape="cylinder",
        score=score,
        confidence=_score_to_confidence(score),
        params=params,
    )


def _make_box_candidate(
    pts: np.ndarray,
    projection_info: Dict[str, Dict[str, float]],
    pca_feats: Dict[str, float],
) -> _ShapeCandidate:
    center = np.mean(pts, axis=0)

    # Since pts are already in PCA frame, box axes align with coordinate axes
    q = pts - center[None, :]
    q_lo = np.quantile(q, 0.02, axis=0)
    q_hi = np.quantile(q, 0.98, axis=0)
    extents = np.maximum(q_hi - q_lo, 1e-6)

    hx, hy, hz = 0.5 * extents[0], 0.5 * extents[1], 0.5 * extents[2]
    q_center = 0.5 * (q_hi + q_lo)
    q0 = q - q_center[None, :]

    dx = np.abs(np.abs(q0[:, 0]) - hx)
    dy = np.abs(np.abs(q0[:, 1]) - hy)
    dz = np.abs(np.abs(q0[:, 2]) - hz)
    d_face = np.minimum(np.minimum(dx, dy), dz)

    obj_scale = float(np.linalg.norm(extents))
    face_rmse = float(np.sqrt(np.mean(d_face ** 2)))
    face_nrmse = float(face_rmse / max(obj_scale, 1e-6))

    tol = 0.06 * np.mean(extents)
    sx_sup = float(np.mean(dx < tol))
    sy_sup = float(np.mean(dy < tol))
    sz_sup = float(np.mean(dz < tol))
    face_support = float(np.mean(sorted([sx_sup, sy_sup, sz_sup], reverse=True)[:2]))

    rect_like = 0
    circle_like = 0
    rect_score = 0.0
    circle_score = 0.0

    for k in ["p12", "p13", "p23"]:
        info = projection_info[k]

        if info["shape_label"] == "rectangle":
            rect_like += 1
        elif info["shape_label"] == "circle":
            circle_like += 1

        rect_score += float(info["rect_score"])
        circle_score += float(info["circle_score"])

    rect_score /= 3.0
    circle_score /= 3.0

    score_fit = float(np.clip(1.0 - face_nrmse / 0.08, 0.0, 1.0))
    score_face = float(np.clip(face_support / 0.8, 0.0, 1.0))
    score_proj = float(np.clip(rect_like / 2.0, 0.0, 1.0))
    penalty_circle = float(np.clip(circle_like / 2.0, 0.0, 1.0))

    score = (
        0.42 * score_fit +
        0.28 * score_face +
        0.20 * score_proj +
        0.06 * rect_score
    ) - 0.18 * penalty_circle - 0.08 * max(circle_score - 0.65, 0.0)

    score = float(np.clip(score, 0.0, 1.0))

    params = {
        "sx": float(extents[0]),
        "sy": float(extents[1]),
        "sz": float(extents[2]),
        "center": center.astype(np.float32),
        "rotation": np.eye(3, dtype=np.float32),
        "fit_score": float(score_fit),
        "face_rmse": float(face_rmse),
        "face_nrmse": float(face_nrmse),
        "face_support": float(face_support),
        "rect_like_count": int(rect_like),
        "circle_like_count": int(circle_like),
        "rect_score_mean": float(rect_score),
        "circle_score_mean": float(circle_score),
        "score_total": float(score),
    }

    return _ShapeCandidate(
        shape="box",
        score=score,
        confidence=_score_to_confidence(score),
        params=params,
    )


def _apply_prior_bias(cand: _ShapeCandidate, prior_shape: str) -> _ShapeCandidate:
    score = float(cand.score)

    if prior_shape == "sphere":
        if cand.shape == "sphere":
            score += 0.10
        elif cand.shape == "box":
            score -= 0.08
        elif cand.shape == "cylinder":
            score += 0.02

    elif prior_shape == "cylinder":
        if cand.shape == "cylinder":
            score += 0.10
        elif cand.shape == "sphere":
            score -= 0.04
        elif cand.shape == "box":
            score -= 0.06

    elif prior_shape == "box":
        if cand.shape == "box":
            score += 0.10
        elif cand.shape == "sphere":
            score -= 0.08
        elif cand.shape == "cylinder":
            score -= 0.02

    score = float(np.clip(score, 0.0, 1.0))
    return _ShapeCandidate(
        shape=cand.shape,
        score=score,
        confidence=_score_to_confidence(score),
        params=dict(cand.params),
    )


# ============================================================
# PCA frame
# ============================================================

def _transform_points_to_pca_frame(
    pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    center = np.mean(pts, axis=0)
    X = pts - center[None, :]

    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # ensure right-handed basis
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1.0

    R_pca = eigvecs.astype(np.float64)  # columns = principal axes
    pts_local = X @ R_pca

    e1 = float(max(eigvals[0], 1e-12))
    e2 = float(max(eigvals[1], 1e-12))
    e3 = float(max(eigvals[2], 1e-12))

    pca_feats = {
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "iso_ratio": float(e1 / max(e3, 1e-6)),
        "axis_ratio": float(e1 / max(e2, 1e-6)),
        "mid_ratio": float(e2 / max(e3, 1e-6)),
    }

    return pts_local, R_pca, center.astype(np.float64), pca_feats


def _convert_candidate_params_back_to_world(
    cand: _ShapeCandidate,
    R_pca: np.ndarray,
    center_world: np.ndarray,
) -> Dict[str, Any]:
    params = dict(cand.params)

    if cand.shape == "sphere":
        if "center" in params:
            center_local = np.asarray(params["center"], dtype=np.float64).reshape(3)
            center_w = center_world + R_pca @ center_local
            params["center"] = center_w.astype(np.float32)

    elif cand.shape == "cylinder":
        if "axis" in params:
            axis_local = np.asarray(params["axis"], dtype=np.float64).reshape(3)
            axis_world = R_pca @ axis_local
            axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)
            params["axis"] = axis_world.astype(np.float32)

        if "center" in params:
            center_local = np.asarray(params["center"], dtype=np.float64).reshape(3)
            center_w = center_world + R_pca @ center_local
            params["center"] = center_w.astype(np.float32)

    elif cand.shape == "box":
        if "rotation" in params:
            R_box_local = np.asarray(params["rotation"], dtype=np.float64)
            R_box_world = R_pca @ R_box_local
            params["rotation"] = R_box_world.astype(np.float32)

        if "center" in params:
            center_local = np.asarray(params["center"], dtype=np.float64).reshape(3)
            center_w = center_world + R_pca @ center_local
            params["center"] = center_w.astype(np.float32)

    return params


# ============================================================
# Projection fitting pipeline
# ============================================================

def infer_shape_by_projection_fitting(
    pts_pca: np.ndarray,
    radius_cv_thresh: float = 0.12,
    radius_range_ratio_thresh: float = 0.22,
) -> Dict[str, Any]:
    """
    Based on PCA projection fitting:
    1) fit circle and rectangle on 3 projections
    2) >= 2 circular projections => sphere
    3) == 1 circular projection => cylinder candidate
    4) cylinder candidate => radius profile analysis along the projection normal
       - constant radius => cylinder
       - obvious change => sphere
    5) 0 circular projections => box
    """
    projection_info = _analyze_three_pca_projections(pts_pca)

    circle_views = []
    rect_views = []
    ambiguous_views = []

    for name, info in projection_info.items():
        if info["shape_label"] == "circle":
            circle_views.append(name)
        elif info["shape_label"] == "rectangle":
            rect_views.append(name)
        else:
            ambiguous_views.append(name)

    if len(circle_views) >= 2:
        return {
            "shape": "sphere",
            "confidence": "high" if len(circle_views) == 3 else "medium",
            "projection_info": projection_info,
            "circle_views": circle_views,
            "rect_views": rect_views,
            "axis_analysis": None,
        }

    if len(circle_views) == 0:
        return {
            "shape": "box",
            "confidence": "medium" if len(rect_views) >= 2 else "low",
            "projection_info": projection_info,
            "circle_views": circle_views,
            "rect_views": rect_views,
            "axis_analysis": None,
        }

    # exactly one circle-like projection => cylinder candidate
    circle_view = circle_views[0]
    axis_name = _projection_to_axis_name(circle_view)
    axis_vec = _axis_name_to_vector(axis_name)

    profile = _compute_radius_profile_along_axis(
        pts=pts_pca,
        axis=axis_vec,
        num_slices=9,
        min_points_per_slice=20,
    )

    if profile["valid"] < 0.5:
        final_shape = "cylinder"
        confidence = "low"
    else:
        if (
            profile["radius_cv"] < radius_cv_thresh and
            profile["radius_range_ratio"] < radius_range_ratio_thresh
        ):
            final_shape = "cylinder"
            confidence = "high"
        else:
            final_shape = "sphere"
            confidence = "medium"

    return {
        "shape": final_shape,
        "confidence": confidence,
        "projection_info": projection_info,
        "circle_views": circle_views,
        "rect_views": rect_views,
        "axis_analysis": {
            "axis_name": axis_name,
            "axis_vector": axis_vec.astype(np.float32),
            "radius_cv": float(profile["radius_cv"]),
            "radius_range_ratio": float(profile["radius_range_ratio"]),
            "mean_radius": float(profile["mean_radius"]),
        },
    }


def _analyze_three_pca_projections(
    pts_pca: np.ndarray
) -> Dict[str, Dict[str, float]]:
    p12 = pts_pca[:, [0, 1]]
    p13 = pts_pca[:, [0, 2]]
    p23 = pts_pca[:, [1, 2]]

    proj = {
        "p12": _classify_projection_shape(p12),
        "p13": _classify_projection_shape(p13),
        "p23": _classify_projection_shape(p23),
    }
    return proj


def _projection_to_axis_name(proj_name: str) -> str:
    # if projection in p12 plane is circular, normal is z
    if proj_name == "p12":
        return "z"
    elif proj_name == "p13":
        return "y"
    elif proj_name == "p23":
        return "x"
    else:
        raise ValueError(f"Unknown projection name: {proj_name}")


def _axis_name_to_vector(axis_name: str) -> np.ndarray:
    if axis_name == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif axis_name == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif axis_name == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        raise ValueError(f"Unknown axis name: {axis_name}")


# ============================================================
# 3D sphere fit
# ============================================================

def _fit_sphere_least_squares(
    pts: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    if pts.shape[0] < 4:
        return None, None, None

    A = np.hstack([2.0 * pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    b = np.sum(pts ** 2, axis=1)

    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None

    center = x[:3]
    d = x[3]
    radius_sq = float(np.sum(center ** 2) + d)

    if radius_sq <= 1e-10:
        return None, None, None

    radius = float(np.sqrt(radius_sq))
    dist = np.linalg.norm(pts - center[None, :], axis=1)
    residual = dist - radius
    residual_std = float(np.std(residual))

    return center.astype(np.float64), radius, residual_std


# ============================================================
# Projection fitting
# ============================================================

def _classify_projection_shape(points_2d: np.ndarray) -> Dict[str, float]:
    circle = _fit_circle_2d(points_2d)
    rect = _fit_rectangle_2d(points_2d)

    result = {
        "circle_score": float(circle["score"]),
        "rect_score": float(rect["score"]),
        "circle_nrmse": float(circle["nrmse"]),
        "rect_nrmse": float(rect["nrmse"]),
        "shape_label": "unknown",
        "shape_confidence": 0.0,
        "circle_radius": float(circle["radius"]),
        "rect_iou": float(rect["iou"]),
        "valid": 1.0 if (circle["valid"] > 0.5 or rect["valid"] > 0.5) else 0.0,
    }

    if circle["valid"] < 0.5 and rect["valid"] < 0.5:
        return result

    diff = float(circle["score"] - rect["score"])

    if diff > 0.08:
        result["shape_label"] = "circle"
        result["shape_confidence"] = float(np.clip(diff, 0.0, 1.0))
    elif diff < -0.08:
        result["shape_label"] = "rectangle"
        result["shape_confidence"] = float(np.clip(-diff, 0.0, 1.0))
    else:
        result["shape_label"] = "ambiguous"
        result["shape_confidence"] = float(np.clip(abs(diff), 0.0, 1.0))

    return result


def _fit_circle_2d(points_2d: np.ndarray) -> Dict[str, float]:
    out = {
        "valid": 0.0,
        "cx": 0.0,
        "cy": 0.0,
        "radius": 0.0,
        "rmse": 1e9,
        "nrmse": 1e9,
        "score": 0.0,
    }

    if points_2d is None or len(points_2d) < 8:
        return out

    pts = np.asarray(points_2d, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 8:
        return out

    x = pts[:, 0]
    y = pts[:, 1]
    A = np.stack([2.0 * x, 2.0 * y, np.ones_like(x)], axis=1)
    b = x ** 2 + y ** 2

    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return out

    cx, cy, c = sol
    r2 = float(cx * cx + cy * cy + c)
    if r2 <= 1e-10:
        return out

    r = float(np.sqrt(r2))
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    err = np.abs(d - r)

    rmse = float(np.sqrt(np.mean(err ** 2)))
    scale = max(r, 1e-6)
    nrmse = float(rmse / scale)

    # higher means more circle-like
    score = float(np.clip(1.0 - nrmse / 0.18, 0.0, 1.0))

    out.update({
        "valid": 1.0,
        "cx": float(cx),
        "cy": float(cy),
        "radius": float(r),
        "rmse": rmse,
        "nrmse": nrmse,
        "score": score,
    })
    return out


def _fit_rectangle_2d(points_2d: np.ndarray) -> Dict[str, float]:
    out = {
        "valid": 0.0,
        "rmse": 1e9,
        "nrmse": 1e9,
        "score": 0.0,
        "w": 0.0,
        "h": 0.0,
        "angle": 0.0,
        "iou": 0.0,
    }

    if points_2d is None or len(points_2d) < 10:
        return out

    img = _rasterize_2d(points_2d)
    if img is None:
        return out

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return out

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 1.0:
        return out

    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), angle = rect
    rw = float(max(rw, 1e-6))
    rh = float(max(rh, 1e-6))
    box = cv2.boxPoints(rect).astype(np.int32)

    x, y, w, h = cv2.boundingRect(contour)
    W = max(x + w + 4, 8)
    H = max(y + h + 4, 8)

    mask_contour = np.zeros((H, W), dtype=np.uint8)
    mask_box = np.zeros((H, W), dtype=np.uint8)

    cv2.drawContours(mask_contour, [contour], -1, 255, thickness=-1)
    cv2.drawContours(mask_box, [box], -1, 255, thickness=-1)

    inter = int(np.logical_and(mask_contour > 0, mask_box > 0).sum())
    union = int(np.logical_or(mask_contour > 0, mask_box > 0).sum())
    iou = float(inter / max(union, 1))

    rmse = float(1.0 - iou)
    nrmse = rmse
    score = float(np.clip((iou - 0.45) / 0.45, 0.0, 1.0))

    out.update({
        "valid": 1.0,
        "rmse": rmse,
        "nrmse": nrmse,
        "score": score,
        "w": rw,
        "h": rh,
        "angle": float(angle),
        "iou": iou,
    })
    return out


def _rasterize_2d(points_2d: np.ndarray, res: float = 0.003) -> Optional[np.ndarray]:
    if points_2d is None or points_2d.shape[0] < 10:
        return None

    pts = points_2d[np.isfinite(points_2d).all(axis=1)]
    if pts.shape[0] < 10:
        return None

    mn = pts.min(axis=0)
    ptsi = ((pts - mn[None, :]) / max(res, 1e-6)).astype(np.int32)

    w = int(ptsi[:, 0].max()) + 5
    h = int(ptsi[:, 1].max()) + 5

    if w <= 2 or h <= 2 or w > 4096 or h > 4096:
        return None

    img = np.zeros((h, w), dtype=np.uint8)

    for x, y in ptsi:
        cv2.circle(img, (int(x), int(y)), 1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img


# ============================================================
# Axis radius profile analysis
# ============================================================

def _compute_radius_profile_along_axis(
    pts: np.ndarray,
    axis: np.ndarray,
    num_slices: int = 9,
    min_points_per_slice: int = 20,
) -> Dict[str, float]:
    out = {
        "valid": 0.0,
        "radius_cv": 1e9,
        "radius_range_ratio": 1e9,
        "mean_radius": 0.0,
    }

    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    center = np.mean(pts, axis=0)
    v = pts - center[None, :]

    proj = v @ axis
    radial = v - np.outer(proj, axis)
    radii = np.linalg.norm(radial, axis=1)

    if len(radii) < 50:
        return out

    pmin, pmax = np.quantile(proj, [0.05, 0.95])
    if (pmax - pmin) < 1e-8:
        return out

    bins = np.linspace(pmin, pmax, num_slices + 1)

    r_meds = []
    for i in range(num_slices):
        if i == num_slices - 1:
            mask = (proj >= bins[i]) & (proj <= bins[i + 1])
        else:
            mask = (proj >= bins[i]) & (proj < bins[i + 1])

        if np.count_nonzero(mask) < min_points_per_slice:
            continue

        r_med = float(np.median(radii[mask]))
        r_meds.append(r_med)

    if len(r_meds) < 4:
        return out

    r_meds = np.asarray(r_meds, dtype=np.float64)
    mean_r = float(np.mean(r_meds))
    std_r = float(np.std(r_meds))
    cv = float(std_r / max(mean_r, 1e-8))
    range_ratio = float((r_meds.max() - r_meds.min()) / max(mean_r, 1e-8))

    out.update({
        "valid": 1.0,
        "radius_cv": cv,
        "radius_range_ratio": range_ratio,
        "mean_radius": mean_r,
    })
    return out


# ============================================================
# Debug visualization
# ============================================================

def save_projection_bundle_debug_with_text(
    projection_points_2d: Dict[str, np.ndarray],
    projection_info: Dict[str, Dict[str, float]],
    save_dir: str,
    prefix: str = "geom_debug",
    img_size: int = 512,
):
    os.makedirs(save_dir, exist_ok=True)

    saved = {}
    for name, p2 in projection_points_2d.items():
        img = _draw_projection_points(
            p2,
            img_size=img_size,
            point_radius=1,
        )

        info = projection_info.get(name, {})
        text1 = f"{name.upper()} valid={info.get('valid', 0.0):.1f}"
        text2 = f"circle={info.get('circle_score', 0.0):.3f} rect={info.get('rect_score', 0.0):.3f}"
        text3 = f"cnrmse={info.get('circle_nrmse', 0.0):.3f} rnrmse={info.get('rect_nrmse', 0.0):.3f}"
        text4 = f"label={info.get('shape_label', 'unknown')}"
        text5 = f"conf={info.get('shape_confidence', 0.0):.3f}"

        cv2.putText(img, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, text3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, text4, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, text5, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(save_dir, f"{prefix}_{name}.png")
        cv2.imwrite(out_path, img)
        saved[name] = out_path

    return saved


def save_xyz_projection_debug(
    pts: np.ndarray,
    save_dir: str,
    prefix: str = "geom_debug",
    img_size: int = 512,
    point_radius: int = 1,
):
    os.makedirs(save_dir, exist_ok=True)

    views = {
        "xy": pts[:, [0, 1]],
        "xz": pts[:, [0, 2]],
        "yz": pts[:, [1, 2]],
    }

    saved = {}
    for name, p2 in views.items():
        img = _draw_projection_points(
            p2,
            img_size=img_size,
            point_radius=point_radius,
        )
        out_path = os.path.join(save_dir, f"{prefix}_{name}.png")
        cv2.imwrite(out_path, img)
        saved[name] = out_path

    return saved


def _draw_projection_points(
    points_2d: np.ndarray,
    img_size: int = 512,
    point_radius: int = 1,
    margin: int = 20,
) -> np.ndarray:
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    if points_2d is None or points_2d.shape[0] == 0:
        return img

    pts = points_2d[np.isfinite(points_2d).all(axis=1)]
    if pts.shape[0] == 0:
        return img

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = mx - mn
    span = np.maximum(span, 1e-6)

    scale = min(
        (img_size - 2 * margin) / span[0],
        (img_size - 2 * margin) / span[1],
    )

    pts_img = (pts - mn[None, :]) * scale
    pts_img[:, 0] += margin
    pts_img[:, 1] += margin
    pts_img[:, 1] = img_size - pts_img[:, 1]

    pts_img = np.round(pts_img).astype(np.int32)

    for x, y in pts_img:
        if 0 <= x < img_size and 0 <= y < img_size:
            cv2.circle(img, (x, y), point_radius, (255, 255, 255), -1)

    x0, y0 = pts_img.min(axis=0)
    x1, y1 = pts_img.max(axis=0)
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)

    return img


def save_radius_profile_debug(
    pts: np.ndarray,
    axis: np.ndarray,
    save_path: str,
    num_slices: int = 9,
    min_points_per_slice: int = 20,
):
    import matplotlib.pyplot as plt

    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    center = np.mean(pts, axis=0)

    v = pts - center[None, :]
    proj = v @ axis
    radial = v - np.outer(proj, axis)
    radii = np.linalg.norm(radial, axis=1)

    pmin, pmax = np.quantile(proj, [0.05, 0.95])
    bins = np.linspace(pmin, pmax, num_slices + 1)

    xs = []
    ys = []

    for i in range(num_slices):
        if i == num_slices - 1:
            mask = (proj >= bins[i]) & (proj <= bins[i + 1])
        else:
            mask = (proj >= bins[i]) & (proj < bins[i + 1])

        if np.count_nonzero(mask) < min_points_per_slice:
            continue

        x_mid = 0.5 * (bins[i] + bins[i + 1])
        r_med = float(np.median(radii[mask]))
        xs.append(x_mid)
        ys.append(r_med)

    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("axis position")
    plt.ylabel("median radius")
    plt.title("Radius profile along axis")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# Point preprocessing
# ============================================================

def _remove_invalid_points(pts: np.ndarray) -> np.ndarray:
    if pts is None or pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    mask = np.isfinite(pts).all(axis=1)
    return pts[mask]


def _robust_trim_points(
    pts: np.ndarray,
    keep_ratio: float = 0.97,
) -> np.ndarray:
    if pts.shape[0] < 30:
        return pts

    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med[None, :], axis=1)

    thresh = np.quantile(d, keep_ratio)
    mask = d <= thresh

    trimmed = pts[mask]
    if trimmed.shape[0] < 20:
        return pts
    return trimmed


def _estimate_voxel_size(pts: np.ndarray) -> float:
    p05 = np.percentile(pts, 5, axis=0)
    p95 = np.percentile(pts, 95, axis=0)
    ext = np.linalg.norm(p95 - p05)
    return float(np.clip(ext / 80.0, 0.002, 0.008))


def _voxel_downsample_numpy(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts

    grid = np.floor(pts / max(voxel_size, 1e-6)).astype(np.int64)
    _, unique_idx = np.unique(grid, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    return pts[unique_idx]


def _object_scale(pts: np.ndarray) -> float:
    p05 = np.percentile(pts, 5, axis=0)
    p95 = np.percentile(pts, 95, axis=0)
    return float(np.linalg.norm(p95 - p05))


# ============================================================
# Helpers
# ============================================================

def _keep_previous(prev_geom) -> UpdatedGeometry:
    if isinstance(prev_geom, UpdatedGeometry):
        return prev_geom

    return UpdatedGeometry(
        shape=prev_geom.shape,
        confidence=prev_geom.confidence,
        allow_upgrade=prev_geom.allow_upgrade,
        allow_downgrade=prev_geom.allow_downgrade,
        params=prev_geom.params,
    )


def _choose_best_with_prior(
    candidates: List[_ShapeCandidate],
    prev_shape: str,
    score_margin: float = 0.06,
) -> _ShapeCandidate:
    candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else candidates[0]

    if abs(best.score - second.score) < score_margin:
        for c in candidates:
            if c.shape == prev_shape:
                return c
    return best


def _score_to_confidence(score: float) -> str:
    if score > 0.80:
        return "high"
    elif score > 0.60:
        return "medium"
    else:
        return "low"


# ============================================================
# Mesh builder
# ============================================================

def build_updated_geometry_mesh(
    geom,
    inflate_ratio: float = 1.05,
    min_extent: float = 0.01,
    cylinder_sections: int = 32,
) -> trimesh.Trimesh:
    """
    Build geometry mesh from UpdatedGeometry.

    Supported:
      - box
      - sphere
      - cylinder
    """
    shape = geom.shape.lower()
    params = geom.params

    if shape == "box":
        sx = max(params["sx"] * inflate_ratio, min_extent)
        sy = max(params["sy"] * inflate_ratio, min_extent)
        sz = max(params["sz"] * inflate_ratio, min_extent)
        return trimesh.creation.box(extents=(sx, sy, sz))

    elif shape == "sphere":
        diameter = max(params.get("diameter", 0.0) * inflate_ratio, min_extent)
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
        raise ValueError(f"Unsupported updated geometry shape: {geom.shape}")