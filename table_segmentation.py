# table_segmentation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import open3d as o3d


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    return v / (np.linalg.norm(v) + 1e-12)


@dataclass
class TableSegConfig:
    # preprocess
    voxel_size: float = 0.005
    remove_stat_outlier: bool = True
    nb_neighbors: int = 30
    std_ratio: float = 2.0

    # plane ransac
    distance_threshold: float = 0.008
    num_iterations: int = 2000
    min_inlier_ratio: float = 0.20

    # remove band
    table_thickness: float = 0.008

    # optional: if you know "up" in THIS coordinate system, you can keep only above
    up_vector: Optional[np.ndarray] = None   # e.g. np.array([0,0,1]) in world; in camera often unknown
    keep_only_above_plane: bool = False
    max_angle_deg: float = 25.0


def preprocess_pcd(pcd: o3d.geometry.PointCloud, cfg: TableSegConfig) -> o3d.geometry.PointCloud:
    work = pcd
    if cfg.voxel_size and cfg.voxel_size > 0:
        work = work.voxel_down_sample(cfg.voxel_size)
    if cfg.remove_stat_outlier and len(work.points) > 50:
        work, _ = work.remove_statistical_outlier(cfg.nb_neighbors, cfg.std_ratio)
    return work


def fit_plane_from_pcd(
    pcd: o3d.geometry.PointCloud,
    cfg: TableSegConfig
) -> Tuple[Optional[np.ndarray], Optional[float], Dict[str, Any]]:
    """
    返回 unit normal n 和 d，使得 n·x + d = 0
    """
    if len(pcd.points) < 200:
        return None, None, {"status": "too_few_points"}

    work = preprocess_pcd(pcd, cfg)
    if len(work.points) < 200:
        return None, None, {"status": "too_few_points_after_preprocess"}

    plane_model, inliers = work.segment_plane(
        distance_threshold=cfg.distance_threshold,
        ransac_n=3,
        num_iterations=cfg.num_iterations,
    )
    a, b, c, d_raw = plane_model
    n_raw = np.array([a, b, c], dtype=float)
    n_norm = np.linalg.norm(n_raw) + 1e-12
    n = n_raw / n_norm
    d = float(d_raw / n_norm)

    inlier_ratio = len(inliers) / max(len(work.points), 1)
    if inlier_ratio < cfg.min_inlier_ratio:
        return None, None, {"status": "inlier_ratio_too_small", "inlier_ratio": float(inlier_ratio)}

    # optional normal constraint
    if cfg.up_vector is not None:
        up = _normalize(cfg.up_vector)
        cosang = float(np.clip(np.abs(np.dot(n, up)), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cosang)))
        if ang > cfg.max_angle_deg:
            return None, None, {"status": "plane_rejected_by_normal", "angle_deg": ang, "inlier_ratio": float(inlier_ratio)}
        # make n point to up for consistent "above"
        if float(np.dot(n, up)) < 0:
            n, d = -n, -d

    return n, d, {
        "status": "ok",
        "inlier_ratio": float(inlier_ratio),
        "n_unit": n.tolist(),
        "d_unit": float(d),
    }


def remove_table_points(
    pcd: o3d.geometry.PointCloud,
    n: np.ndarray,
    d: float,
    cfg: TableSegConfig
) -> o3d.geometry.PointCloud:
    """
    在同一坐标系下用平面删桌面：默认删厚度带（abs），如果提供 up_vector 且 keep_only_above_plane=True 则只保留上方
    """
    n = _normalize(n)
    pts = np.asarray(pcd.points)
    s = pts @ n + float(d)

    if cfg.keep_only_above_plane and cfg.up_vector is not None:
        keep = s > cfg.table_thickness
    else:
        keep = np.abs(s) > cfg.table_thickness

    idx = np.where(keep)[0]
    return pcd.select_by_index(idx)


def segment_objects_from_first_frame(
    pcd_cam: o3d.geometry.PointCloud,
    cfg: Optional[TableSegConfig] = None
) -> Tuple[o3d.geometry.PointCloud, Tuple[np.ndarray, float], Dict[str, Any]]:
    """
    第一帧：拟合桌面并删除桌面，返回物体点云（相机坐标） + 平面模型(n,d)
    """
    cfg = cfg or TableSegConfig()
    n, d, info = fit_plane_from_pcd(pcd_cam, cfg)
    if n is None:
        return o3d.geometry.PointCloud(pcd_cam), (np.zeros(3), 0.0), {"status": "fail", **info}

    obj = remove_table_points(pcd_cam, n, d, cfg)
    return obj, (n, d), {"status": "ok", **info, "points_in": int(len(pcd_cam.points)), "points_obj": int(len(obj.points))}