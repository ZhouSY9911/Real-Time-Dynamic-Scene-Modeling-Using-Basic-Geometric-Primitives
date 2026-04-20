from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import os
import numpy as np
import cv2
import open3d as o3d


# =========================================================
# Config
# =========================================================
@dataclass
class SecondarySplitConfig:
    # 是否启用点云二次分割
    enable_pointcloud_secondary_split: bool = True

    # ---------- per-cluster denoise ----------
    cluster_denoise_voxel_size: float = 0.003
    cluster_denoise_nb_neighbors: int = 20
    cluster_denoise_std_ratio: float = 1.5
    cluster_keep_largest_only: bool = True

    # ---------- trigger ----------
    secondary_split_min_points: int = 1800
    secondary_split_min_extent_max: float = 0.08

    # ---------- KNN density ----------
    knn_k: int = 20
    neck_density_percentile: float = 20.0
    neck_ratio_thresh: float = 0.35
    low_density_keepout_margin: float = 1.15

    # ---------- graph clustering after removing low-density bridge ----------
    component_neighbor_radius: float = 0.018
    component_min_points: int = 180
    split_min_cluster_ratio: float = 0.15
    split_min_center_dist: float = 0.025
    split_min_gap_dist: float = 0.006
    split_min_mask_area_px: int = 180

    # ---------- quality ----------
    reject_slender_ratio_1: float = 8.0
    reject_slender_ratio_2: float = 6.0
    min_largest_component_ratio: float = 0.80
    min_cluster_density: float = 8000.0
    split_score_margin: float = 2.0

    # ---------- recursive ----------
    enable_recursive_split: bool = True
    recursive_split_max_depth: int = 2
    recursive_split_min_points: int = 800
    recursive_split_min_extent_max: float = 0.05

    # ---------- depth support ----------
    enable_depth_assisted_split: bool = True

    # 深度连通性：相邻像素深度差阈值（米）
    depth_connect_thresh: float = 0.010

    # 深度边缘
    depth_grad_ksize: int = 3
    depth_boundary_band_dilate: int = 7
    depth_boundary_min_score: float = 0.08

    # 深度有效范围
    min_valid_depth: float = 1e-6
    max_valid_depth: float = 10.0

    # 深度质量
    min_depth_connected_ratio: float = 0.55

    # watershed 支持
    enable_depth_watershed_support: bool = True
    depth_watershed_min_region_area: int = 200
    depth_watershed_peak_ratio: float = 0.45

    # 深度融合权重
    depth_boundary_weight: float = 2.5
    depth_connectivity_weight: float = 1.5
    depth_watershed_weight: float = 1.0

    # 深度预处理
    enable_depth_blur: bool = True
    depth_blur_ksize: int = 5
    depth_blur_sigma: float = 1.0


# =========================================================
# IO / basic point cloud utils
# =========================================================
def load_pcd_points_and_colors(path: str) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.io.read_point_cloud(path)

    pts = np.asarray(pcd.points).astype(np.float32)
    cols = np.asarray(pcd.colors).astype(np.float32)

    if cols.shape[0] != pts.shape[0]:
        cols = np.zeros((pts.shape[0], 3), dtype=np.float32)

    return pts, cols


def save_pcd_points_and_colors(path: str, points: np.ndarray, colors: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors is not None and colors.shape[0] == points.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    o3d.io.write_point_cloud(path, pcd)


def voxel_downsample_points_and_colors(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return points, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors is not None and colors.shape[0] == points.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    pcd = pcd.voxel_down_sample(voxel_size)

    pts_ds = np.asarray(pcd.points).astype(np.float32)
    cols_ds = np.asarray(pcd.colors).astype(np.float32)

    if cols_ds.shape[0] != pts_ds.shape[0]:
        cols_ds = np.zeros((pts_ds.shape[0], 3), dtype=np.float32)

    return pts_ds, cols_ds


# =========================================================
# Denoise / Quality
# =========================================================
def denoise_points_and_colors(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float = 0.003,
    nb_neighbors: int = 20,
    std_ratio: float = 1.5,
    keep_largest_only: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return points, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and colors.shape[0] == points.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    try:
        if len(pcd.points) >= nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
    except Exception:
        pass

    if keep_largest_only and len(pcd.points) > 0:
        try:
            labels = np.array(
                pcd.cluster_dbscan(
                    eps=max(voxel_size * 3.0, 0.008),
                    min_points=max(10, min(nb_neighbors, 20)),
                    print_progress=False
                )
            )
            if labels.size > 0 and np.any(labels >= 0):
                valid = labels[labels >= 0]
                counts = np.bincount(valid)
                best_lab = int(np.argmax(counts))
                idx = np.where(labels == best_lab)[0]
                pcd = pcd.select_by_index(idx)
        except Exception:
            pass

    pts = np.asarray(pcd.points).astype(np.float32)
    cols = np.asarray(pcd.colors).astype(np.float32)
    if cols.shape[0] != pts.shape[0]:
        cols = np.zeros((pts.shape[0], 3), dtype=np.float32)

    return pts, cols


def compute_pointcloud_extent(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    pmin = np.percentile(points, 5, axis=0)
    pmax = np.percentile(points, 95, axis=0)
    return (pmax - pmin).astype(np.float32)


def compute_pca_scales(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 10:
        return np.zeros(3, dtype=np.float32)

    c = points.mean(axis=0, keepdims=True)
    X = points - c
    cov = X.T @ X / max(points.shape[0] - 1, 1)
    eigvals, _ = np.linalg.eigh(cov)
    eigvals = np.sort(np.maximum(eigvals, 1e-12))[::-1]
    return np.sqrt(eigvals).astype(np.float32)


def is_bad_shape_cluster(points: np.ndarray, cfg: SecondarySplitConfig) -> bool:
    if points.shape[0] < 20:
        return True

    s = compute_pca_scales(points)
    s1, s2, s3 = float(s[0]), float(s[1]), float(s[2])

    if s1 / (s2 + 1e-6) > cfg.reject_slender_ratio_1 and s2 / (s3 + 1e-6) > cfg.reject_slender_ratio_2:
        return True

    return False


def cluster_density_score(points: np.ndarray) -> float:
    if points.shape[0] < 10:
        return 0.0

    pmin = np.percentile(points, 5, axis=0)
    pmax = np.percentile(points, 95, axis=0)
    vol = float(np.prod(np.maximum(pmax - pmin, 1e-4)))
    return float(points.shape[0]) / vol


def largest_component_ratio(mask: np.ndarray) -> float:
    bin_mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num_labels <= 1:
        return 1.0 if np.any(bin_mask > 0) else 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    total = int(np.sum(areas))
    if total <= 0:
        return 0.0
    return float(np.max(areas)) / float(total)


def project_points_to_mask(
    points_c: np.ndarray,
    K: np.ndarray,
    image_shape: Tuple[int, int],
    dilate_ksize: int = 5,
) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if points_c.shape[0] == 0:
        return mask

    z = points_c[:, 2]
    valid = z > 1e-6
    pts = points_c[valid]
    if pts.shape[0] == 0:
        return mask

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.round(pts[:, 0] * fx / pts[:, 2] + cx).astype(np.int32)
    v = np.round(pts[:, 1] * fy / pts[:, 2] + cy).astype(np.int32)

    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[inside]
    v = v[inside]

    mask[v, u] = 255

    if dilate_ksize > 1:
        kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def project_points_to_pixels(
    points_c: np.ndarray,
    K: np.ndarray,
    image_shape: Tuple[int, int],
):
    h, w = image_shape[:2]

    if points_c.shape[0] == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=bool),
        )

    z = points_c[:, 2]
    valid = z > 1e-6

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.round(points_c[:, 0] * fx / np.maximum(points_c[:, 2], 1e-6) + cx).astype(np.int32)
    v = np.round(points_c[:, 1] * fy / np.maximum(points_c[:, 2], 1e-6) + cy).astype(np.int32)

    inside = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u, v, inside

def split_points_by_2d_masks(
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    image_shape: Tuple[int, int],
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if points.shape[0] == 0:
        return []

    u, v, inside = project_points_to_pixels(points, K, image_shape)
    if np.count_nonzero(inside) == 0:
        return []

    idx_all = np.where(inside)[0]
    uu = u[inside]
    vv = v[inside]

    hit_a = mask_a[vv, uu] > 0
    hit_b = mask_b[vv, uu] > 0

    idx_a = idx_all[hit_a & (~hit_b)]
    idx_b = idx_all[hit_b & (~hit_a)]

    # 对于同时命中的点，按离两个mask中心的距离分配
    overlap_idx = idx_all[hit_a & hit_b]
    if overlap_idx.size > 0:
        ys_a, xs_a = np.where(mask_a > 0)
        ys_b, xs_b = np.where(mask_b > 0)

        if len(xs_a) > 0 and len(xs_b) > 0:
            ca = np.array([xs_a.mean(), ys_a.mean()], dtype=np.float32)
            cb = np.array([xs_b.mean(), ys_b.mean()], dtype=np.float32)

            ov_u = u[overlap_idx].astype(np.float32)
            ov_v = v[overlap_idx].astype(np.float32)
            pa = np.stack([ov_u, ov_v], axis=1)

            da = np.linalg.norm(pa - ca[None, :], axis=1)
            db = np.linalg.norm(pa - cb[None, :], axis=1)

            idx_a = np.concatenate([idx_a, overlap_idx[da <= db]])
            idx_b = np.concatenate([idx_b, overlap_idx[db < da]])

    pts_a = points[idx_a]
    cols_a = colors[idx_a] if colors.shape[0] == points.shape[0] else np.zeros((len(idx_a), 3), np.float32)

    pts_b = points[idx_b]
    cols_b = colors[idx_b] if colors.shape[0] == points.shape[0] else np.zeros((len(idx_b), 3), np.float32)

    out = []
    if pts_a.shape[0] > 0:
        out.append((pts_a, cols_a))
    if pts_b.shape[0] > 0:
        out.append((pts_b, cols_b))
    return out

def min_intercluster_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if points_a.shape[0] == 0 or points_b.shape[0] == 0:
        return 1e9

    na = min(300, points_a.shape[0])
    nb = min(300, points_b.shape[0])

    idx_a = np.random.choice(points_a.shape[0], na, replace=False)
    idx_b = np.random.choice(points_b.shape[0], nb, replace=False)

    A = points_a[idx_a]
    B = points_b[idx_b]

    dists = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return float(np.min(dists))


# =========================================================
# Depth utilities
# =========================================================
def normalize_depth_map_if_needed(depth_map: np.ndarray) -> np.ndarray:
    if depth_map is None:
        return depth_map

    if depth_map.dtype == np.uint16:
        return depth_map.astype(np.float32) / 1000.0

    return depth_map.astype(np.float32)


def validate_depth_map_for_split(
    depth_map: Optional[np.ndarray],
    image_shape: Tuple[int, int],
    allow_none: bool = True,
) -> bool:
    if depth_map is None:
        return bool(allow_none)

    if not isinstance(depth_map, np.ndarray):
        return False

    if depth_map.ndim != 2:
        return False

    h, w = image_shape[:2]
    if depth_map.shape[0] != h or depth_map.shape[1] != w:
        return False

    return True


def preprocess_depth_map(
    depth: Optional[np.ndarray],
    cfg: SecondarySplitConfig,
) -> Optional[np.ndarray]:
    if depth is None:
        return None

    depth = depth.astype(np.float32).copy()

    invalid = (depth < cfg.min_valid_depth) | (depth > cfg.max_valid_depth) | ~np.isfinite(depth)
    depth[invalid] = 0.0

    if cfg.enable_depth_blur:
        k = max(1, int(cfg.depth_blur_ksize))
        if k % 2 == 0:
            k += 1
        depth_blur = cv2.GaussianBlur(
            depth,
            (k, k),
            cfg.depth_blur_sigma
        )
        keep = depth > 0
        depth[keep] = depth_blur[keep]

    return depth


def compute_depth_gradient_map(
    depth: Optional[np.ndarray],
    cfg: SecondarySplitConfig
) -> Optional[np.ndarray]:
    if depth is None:
        return None

    depth = depth.astype(np.float32)
    k = max(1, int(cfg.depth_grad_ksize))
    if k % 2 == 0:
        k += 1

    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=k)
    grad = np.sqrt(gx * gx + gy * gy)
    grad[~np.isfinite(grad)] = 0.0
    return grad.astype(np.float32)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def largest_depth_connected_component_ratio(
    mask: np.ndarray,
    depth: Optional[np.ndarray],
    depth_thresh: float = 0.01,
    min_valid_depth: float = 1e-6,
) -> float:
    if depth is None:
        return largest_component_ratio(mask)

    h, w = mask.shape
    valid = (mask > 0) & np.isfinite(depth) & (depth > min_valid_depth)
    if not np.any(valid):
        return 0.0

    visited = np.zeros((h, w), dtype=bool)
    areas: List[int] = []

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1),
    ]

    ys, xs = np.where(valid)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue

        q = [(sy, sx)]
        visited[sy, sx] = True
        area = 0

        while q:
            y, x = q.pop()
            area += 1
            d0 = float(depth[y, x])

            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if visited[ny, nx]:
                    continue
                if not valid[ny, nx]:
                    continue

                d1 = float(depth[ny, nx])
                if abs(d1 - d0) <= depth_thresh:
                    visited[ny, nx] = True
                    q.append((ny, nx))

        areas.append(area)

    if len(areas) == 0:
        return 0.0

    total = int(np.sum(areas))
    if total <= 0:
        return 0.0

    return float(np.max(areas)) / float(total)


def compute_boundary_band_between_masks(
    mask1: np.ndarray,
    mask2: np.ndarray,
    dilate_ksize: int = 7,
) -> np.ndarray:
    m1 = (mask1 > 0).astype(np.uint8)
    m2 = (mask2 > 0).astype(np.uint8)

    k = max(1, int(dilate_ksize))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)

    d1 = cv2.dilate(m1, kernel, iterations=1)
    d2 = cv2.dilate(m2, kernel, iterations=1)

    band = ((d1 > 0) & (d2 > 0)).astype(np.uint8) * 255
    return band


def compute_depth_boundary_score_between_masks(
    depth: Optional[np.ndarray],
    grad_map: Optional[np.ndarray],
    full_mask: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    cfg: SecondarySplitConfig,
) -> float:
    if depth is None or grad_map is None:
        return 0.0

    band = compute_boundary_band_between_masks(
        mask1, mask2, dilate_ksize=cfg.depth_boundary_band_dilate
    )

    valid_band = (band > 0) & np.isfinite(depth) & (depth > cfg.min_valid_depth)
    if np.count_nonzero(valid_band) < 20:
        return 0.0

    vals_band = grad_map[valid_band]
    if vals_band.size == 0:
        return 0.0

    valid_full = (full_mask > 0) & np.isfinite(depth) & (depth > cfg.min_valid_depth)
    if np.count_nonzero(valid_full) < 20:
        return 0.0

    vals_full = grad_map[valid_full]
    if vals_full.size == 0:
        return 0.0

    mean_band = float(np.mean(vals_band))
    p90_band = float(np.percentile(vals_band, 90))
    mean_full = float(np.mean(vals_full)) + 1e-6

    # 相对评分，比绝对阈值更稳
    score = (0.5 * mean_band + 0.5 * p90_band) / mean_full
    return float(score)


def split_mask_by_depth_watershed(
    depth: Optional[np.ndarray],
    obj_mask: np.ndarray,
    cfg: SecondarySplitConfig,
) -> List[np.ndarray]:
    if depth is None:
        return []

    mask = (obj_mask > 0).astype(np.uint8)
    if np.count_nonzero(mask) < cfg.depth_watershed_min_region_area:
        return []

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    thr = cfg.depth_watershed_peak_ratio * float(dist.max())
    peaks = (dist >= thr).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks, connectivity=8)
    if num_labels <= 2:
        return []

    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_ids = np.where(areas >= cfg.depth_watershed_min_region_area)[0] + 1
    if valid_ids.size < 2:
        return []

    # 取最大的两个峰
    valid_ids = sorted(valid_ids, key=lambda x: stats[x, cv2.CC_STAT_AREA], reverse=True)[:2]

    ys, xs = np.where(mask > 0)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)

    c1 = centroids[valid_ids[0]].astype(np.float32)
    c2 = centroids[valid_ids[1]].astype(np.float32)

    d1 = np.linalg.norm(pts - c1[None, :], axis=1)
    d2 = np.linalg.norm(pts - c2[None, :], axis=1)

    mask1 = np.zeros_like(mask, dtype=np.uint8)
    mask2 = np.zeros_like(mask, dtype=np.uint8)

    pts1 = pts[d1 <= d2].astype(np.int32)
    pts2 = pts[d2 < d1].astype(np.int32)

    mask1[pts1[:, 1], pts1[:, 0]] = 255
    mask2[pts2[:, 1], pts2[:, 0]] = 255

    kernel = np.ones((3, 3), np.uint8)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

    if np.count_nonzero(mask1) < cfg.split_min_mask_area_px:
        return []
    if np.count_nonzero(mask2) < cfg.split_min_mask_area_px:
        return []

    return [mask1, mask2]

def cluster_quality_score(
    points: np.ndarray,
    mask: np.ndarray,
    cfg: SecondarySplitConfig,
    depth: Optional[np.ndarray] = None,
) -> float:
    if points.shape[0] < 50:
        return -1e6

    if is_bad_shape_cluster(points, cfg):
        return -1000.0

    density = cluster_density_score(points)
    if density < cfg.min_cluster_density:
        return -500.0

    conn_ratio_2d = largest_component_ratio(mask)
    if conn_ratio_2d < cfg.min_largest_component_ratio:
        return -300.0

    depth_conn_ratio = 1.0
    if cfg.enable_depth_assisted_split and depth is not None:
        depth_conn_ratio = largest_depth_connected_component_ratio(
            mask=mask,
            depth=depth,
            depth_thresh=cfg.depth_connect_thresh,
            min_valid_depth=cfg.min_valid_depth,
        )
        if depth_conn_ratio < cfg.min_depth_connected_ratio:
            return -250.0

    score = (
        0.01 * float(points.shape[0]) +
        0.0001 * density +
        2.0 * conn_ratio_2d
    )

    if depth is not None and cfg.enable_depth_assisted_split:
        score += cfg.depth_connectivity_weight * depth_conn_ratio

    return float(score)

def split_pointcloud_by_depth_mask_watershed(
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    image_shape: Tuple[int, int],
    obj_mask: np.ndarray,
    depth_map: Optional[np.ndarray],
    cfg: SecondarySplitConfig,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if depth_map is None:
        return []

    sub_masks = split_mask_by_depth_watershed(depth_map, obj_mask, cfg)
    if len(sub_masks) < 2:
        return []

    clusters = split_points_by_2d_masks(
        points=points,
        colors=colors,
        K=K,
        image_shape=image_shape,
        mask_a=sub_masks[0],
        mask_b=sub_masks[1],
    )

    refined = []
    for sub_pts, sub_cols in clusters:
        sub_pts, sub_cols = denoise_points_and_colors(
            sub_pts,
            sub_cols,
            voxel_size=cfg.cluster_denoise_voxel_size,
            nb_neighbors=cfg.cluster_denoise_nb_neighbors,
            std_ratio=cfg.cluster_denoise_std_ratio,
            keep_largest_only=cfg.cluster_keep_largest_only,
        )
        if sub_pts.shape[0] > 0:
            refined.append((sub_pts, sub_cols))

    return refined


# =========================================================
# KNN density based split
# =========================================================
def compute_knn_distance_features(points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
    - kth_dist: 每个点到第k近邻的距离
    - mean_dist: 每个点到前k近邻平均距离
    """
    n = points.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    kth_dist = np.zeros((n,), dtype=np.float32)
    mean_dist = np.zeros((n,), dtype=np.float32)

    kk = min(k + 1, n)

    for i in range(n):
        _, idx, dist2 = kdtree.search_knn_vector_3d(points[i].astype(np.float64), kk)
        if len(dist2) <= 1:
            kth_dist[i] = 1e6
            mean_dist[i] = 1e6
            continue

        dist = np.sqrt(np.asarray(dist2[1:], dtype=np.float32))
        kth_dist[i] = float(dist[-1])
        mean_dist[i] = float(np.mean(dist))

    return kth_dist, mean_dist


def compute_local_density(points: np.ndarray, k: int) -> np.ndarray:
    _, mean_dist = compute_knn_distance_features(points, k)
    density = 1.0 / (mean_dist + 1e-6)
    return density.astype(np.float32)


def compute_neck_score(points: np.ndarray, cfg: SecondarySplitConfig) -> Tuple[float, Dict[str, float]]:
    if points.shape[0] < max(cfg.knn_k + 5, 30):
        return 1.0, {
            "density_low": 0.0,
            "density_med": 0.0,
            "neck_score": 1.0,
        }

    density = compute_local_density(points, cfg.knn_k)
    density_low = float(np.percentile(density, cfg.neck_density_percentile))
    density_med = float(np.median(density))
    neck_score = density_low / (density_med + 1e-6)

    return float(neck_score), {
        "density_low": density_low,
        "density_med": density_med,
        "neck_score": float(neck_score),
    }


def connected_components_from_points_by_radius(
    points: np.ndarray,
    radius: float,
    min_points: int,
) -> List[np.ndarray]:
    n = points.shape[0]
    if n == 0:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    visited = np.zeros((n,), dtype=bool)
    components: List[np.ndarray] = []

    for i in range(n):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        comp = []

        while queue:
            cur = queue.pop()
            comp.append(cur)

            _, idx, _ = kdtree.search_radius_vector_3d(points[cur].astype(np.float64), radius)
            for j in idx:
                if not visited[j]:
                    visited[j] = True
                    queue.append(j)

        comp_idx = np.asarray(comp, dtype=np.int32)
        if comp_idx.size >= min_points:
            components.append(comp_idx)

    components.sort(key=lambda x: x.size, reverse=True)
    return components


def split_pointcloud_by_knn_neck(
    points: np.ndarray,
    colors: np.ndarray,
    cfg: SecondarySplitConfig,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if points.shape[0] == 0:
        return []

    density = compute_local_density(points, cfg.knn_k)
    density_thr = float(np.percentile(density, cfg.neck_density_percentile)) * cfg.low_density_keepout_margin

    keep = density >= density_thr
    kept_idx = np.where(keep)[0]

    if kept_idx.size < max(cfg.component_min_points * 2, 100):
        return []

    kept_points = points[kept_idx]
    kept_colors = colors[kept_idx] if colors.shape[0] == points.shape[0] else np.zeros((kept_idx.size, 3), dtype=np.float32)

    components = connected_components_from_points_by_radius(
        points=kept_points,
        radius=cfg.component_neighbor_radius,
        min_points=cfg.component_min_points,
    )

    if len(components) < 2:
        return []

    clusters: List[Tuple[np.ndarray, np.ndarray]] = []
    for comp in components[:2]:
        sub_pts = kept_points[comp]
        sub_cols = kept_colors[comp]

        sub_pts, sub_cols = denoise_points_and_colors(
            sub_pts,
            sub_cols,
            voxel_size=cfg.cluster_denoise_voxel_size,
            nb_neighbors=cfg.cluster_denoise_nb_neighbors,
            std_ratio=cfg.cluster_denoise_std_ratio,
            keep_largest_only=cfg.cluster_keep_largest_only,
        )

        if sub_pts.shape[0] > 0:
            clusters.append((sub_pts, sub_cols))

    return clusters


# =========================================================
# Split logic
# =========================================================
def choose_best_split_candidate(
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    image_shape: Tuple[int, int],
    cfg: SecondarySplitConfig,
    depth_map: Optional[np.ndarray] = None,
    original_mask: Optional[np.ndarray] = None,
):
    candidates = []

    # candidate A: KNN neck
    c1 = split_pointcloud_by_knn_neck(points, colors, cfg)
    if len(c1) >= 2:
        ret1 = evaluate_split_candidate(
            full_points=points,
            clusters=c1,
            K=K,
            image_shape=image_shape,
            cfg=cfg,
            depth_map=depth_map,
            original_mask=original_mask,
        )
        ok1, gain1 = ret1[0], ret1[1]
        if ok1:
            candidates.append((gain1, c1))

    # candidate B: depth/mask watershed
    if original_mask is not None and depth_map is not None:
        c2 = split_pointcloud_by_depth_mask_watershed(
            points=points,
            colors=colors,
            K=K,
            image_shape=image_shape,
            obj_mask=original_mask,
            depth_map=depth_map,
            cfg=cfg,
        )
        if len(c2) >= 2:
            ret2 = evaluate_split_candidate(
                full_points=points,
                clusters=c2,
                K=K,
                image_shape=image_shape,
                cfg=cfg,
                depth_map=depth_map,
                original_mask=original_mask,
            )
            ok2, gain2 = ret2[0], ret2[1]
            if ok2:
                candidates.append((gain2, c2))

    if len(candidates) == 0:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def is_candidate_for_secondary_split(points: np.ndarray, cfg: SecondarySplitConfig) -> bool:
    if points.shape[0] < cfg.secondary_split_min_points:
        return False

    extent = compute_pointcloud_extent(points)
    if float(np.max(extent)) < cfg.secondary_split_min_extent_max:
        return False

    scales = compute_pca_scales(points)
    s1, s2, s3 = float(scales[0]), float(scales[1]), float(scales[2])

    # 很细长且更像单体的，不做二次分割
    if s1 / (s2 + 1e-6) > 10.0 and s2 / (s3 + 1e-6) < 2.5:
        return False

    return True


def is_candidate_for_recursive_split(points: np.ndarray, cfg: SecondarySplitConfig) -> bool:
    if points.shape[0] < cfg.recursive_split_min_points:
        return False

    extent = compute_pointcloud_extent(points)
    if float(np.max(extent)) < cfg.recursive_split_min_extent_max:
        return False

    scales = compute_pca_scales(points)
    s1, s2, s3 = float(scales[0]), float(scales[1]), float(scales[2])

    if s1 / (s2 + 1e-6) > 10.0 and s2 / (s3 + 1e-6) < 2.5:
        return False

    return True


def evaluate_split_candidate(
    full_points: np.ndarray,
    clusters: List[Tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    image_shape: Tuple[int, int],
    cfg: SecondarySplitConfig,
    depth_map: Optional[np.ndarray] = None,
    depth_grad_map: Optional[np.ndarray] = None,
    original_mask: Optional[np.ndarray] = None,
) -> Tuple[bool, float, Dict[str, Any]]:
    debug: Dict[str, Any] = {}

    if len(clusters) < 2:
        return False, -1e9, {"reason": "less_than_two_clusters"}

    total_n = full_points.shape[0]
    if total_n < cfg.secondary_split_min_points:
        return False, -1e9, {"reason": "too_few_total_points"}

    sizes = [c[0].shape[0] for c in clusters]
    n1, n2 = sizes[0], sizes[1]
    debug["cluster_sizes"] = sizes

    if n1 < cfg.component_min_points or n2 < cfg.component_min_points:
        return False, -1e9, {"reason": "cluster_too_small", "cluster_sizes": sizes}

    if n2 < total_n * cfg.split_min_cluster_ratio:
        return False, -1e9, {"reason": "second_cluster_ratio_too_small", "cluster_sizes": sizes}

    c1 = np.median(clusters[0][0], axis=0)
    c2 = np.median(clusters[1][0], axis=0)
    center_dist = float(np.linalg.norm(c1 - c2))
    debug["center_dist"] = center_dist
    if center_dist < cfg.split_min_center_dist:
        return False, -1e9, {"reason": "center_dist_too_small", "center_dist": center_dist}

    gap_dist = min_intercluster_distance(clusters[0][0], clusters[1][0])
    debug["gap_dist"] = gap_dist
    if gap_dist < cfg.split_min_gap_dist:
        return False, -1e9, {"reason": "gap_dist_too_small", "gap_dist": gap_dist}

    full_mask = project_points_to_mask(full_points, K, image_shape, dilate_ksize=5)
    mask1 = project_points_to_mask(clusters[0][0], K, image_shape, dilate_ksize=5)
    mask2 = project_points_to_mask(clusters[1][0], K, image_shape, dilate_ksize=5)

    area1 = int(np.count_nonzero(mask1))
    area2 = int(np.count_nonzero(mask2))
    debug["mask_area_1"] = area1
    debug["mask_area_2"] = area2

    if area1 < cfg.split_min_mask_area_px:
        return False, -1e9, {"reason": "mask1_area_too_small", "mask_area_1": area1}
    if area2 < cfg.split_min_mask_area_px:
        return False, -1e9, {"reason": "mask2_area_too_small", "mask_area_2": area2}

    score_before = cluster_quality_score(
        full_points, full_mask, cfg, depth=depth_map
    )

    score_after = 0.0
    for pts, _ in clusters[:2]:
        sub_mask = project_points_to_mask(pts, K, image_shape, dilate_ksize=5)
        score_after += cluster_quality_score(
            pts, sub_mask, cfg, depth=depth_map
        )

    gain = score_after - score_before
    debug["score_before"] = float(score_before)
    debug["score_after"] = float(score_after)
    debug["quality_gain"] = float(gain)

    depth_bonus = 0.0
    depth_boundary_score = 0.0
    full_depth_conn = None
    m1_depth_conn = None
    m2_depth_conn = None
    ws_ok, ws_score = False, 0.0

    if cfg.enable_depth_assisted_split and depth_map is not None:
        depth_boundary_score = compute_depth_boundary_score_between_masks(
            depth=depth_map,
            grad_map=depth_grad_map,
            full_mask=full_mask,
            mask1=mask1,
            mask2=mask2,
            cfg=cfg,
        )

        if depth_boundary_score < cfg.depth_boundary_min_score:
            depth_bonus -= cfg.depth_boundary_weight * 0.5
        else:
            depth_bonus += cfg.depth_boundary_weight * depth_boundary_score

        full_depth_conn = largest_depth_connected_component_ratio(
            mask=full_mask,
            depth=depth_map,
            depth_thresh=cfg.depth_connect_thresh,
            min_valid_depth=cfg.min_valid_depth,
        )
        m1_depth_conn = largest_depth_connected_component_ratio(
            mask=mask1,
            depth=depth_map,
            depth_thresh=cfg.depth_connect_thresh,
            min_valid_depth=cfg.min_valid_depth,
        )
        m2_depth_conn = largest_depth_connected_component_ratio(
            mask=mask2,
            depth=depth_map,
            depth_thresh=cfg.depth_connect_thresh,
            min_valid_depth=cfg.min_valid_depth,
        )

        conn_gain = (m1_depth_conn + m2_depth_conn) - full_depth_conn
        depth_bonus += cfg.depth_connectivity_weight * conn_gain

        if original_mask is None:
            original_mask = full_mask

        #ws_ok, ws_score = compute_depth_watershed_support(
        #    depth=depth_map,
        #    obj_mask=original_mask,
        #    cfg=cfg,
        #)
        #if ws_ok:
        #    depth_bonus += cfg.depth_watershed_weight * ws_score

    final_gain = gain + depth_bonus
    accept = final_gain > cfg.split_score_margin

    debug["depth_boundary_score"] = float(depth_boundary_score)
    debug["full_depth_conn"] = None if full_depth_conn is None else float(full_depth_conn)
    debug["mask1_depth_conn"] = None if m1_depth_conn is None else float(m1_depth_conn)
    debug["mask2_depth_conn"] = None if m2_depth_conn is None else float(m2_depth_conn)
    debug["watershed_ok"] = bool(ws_ok)
    debug["watershed_score"] = float(ws_score)
    debug["depth_bonus"] = float(depth_bonus)
    debug["final_gain"] = float(final_gain)
    debug["accepted"] = bool(accept)

    return accept, float(final_gain), debug


def recursive_split_pointcloud(
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    image_shape: Tuple[int, int],
    cfg: SecondarySplitConfig,
    depth_map: Optional[np.ndarray] = None,
    depth_grad_map: Optional[np.ndarray] = None,
    original_mask: Optional[np.ndarray] = None,
    depth: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if points.shape[0] == 0:
        return []

    if depth >= cfg.recursive_split_max_depth:
        return [(points, colors)]

    if not is_candidate_for_recursive_split(points, cfg):
        return [(points, colors)]

    neck_score, _ = compute_neck_score(points, cfg)
    #if neck_score > cfg.neck_ratio_thresh:
    #    return [(points, colors)]

    clusters = choose_best_split_candidate(
    points=points,
    colors=colors,
    K=K,
    image_shape=image_shape,
    cfg=cfg,
    depth_map=depth_map,
    original_mask=original_mask,
    )

    ret = evaluate_split_candidate(
        full_points=points,
        clusters=clusters,
        K=K,
        image_shape=image_shape,
        cfg=cfg,
        depth_map=depth_map,
        depth_grad_map=depth_grad_map,
        original_mask=original_mask,
    )
    ok = ret[0]
    gain = ret[1]
    if not ok:
        return [(points, colors)]

    results: List[Tuple[np.ndarray, np.ndarray]] = []
    for sub_pts, sub_cols in clusters[:2]:
        child_mask = project_points_to_mask(
            sub_pts, K, image_shape, dilate_ksize=5
        )

        child_results = recursive_split_pointcloud(
            points=sub_pts,
            colors=sub_cols,
            K=K,
            image_shape=image_shape,
            cfg=cfg,
            depth_map=depth_map,
            depth_grad_map=depth_grad_map,
            original_mask=child_mask,
            depth=depth + 1,
        )
        results.extend(child_results)

    if len(results) == 0:
        return [(points, colors)]

    return results


# =========================================================
# Main object refine
# =========================================================
def refine_one_object_by_pointcloud_split(
    obj_path: str,
    mask_path: str,
    K: np.ndarray,
    image_shape: Tuple[int, int],
    save_dir: str,
    out_prefix: str,
    cfg: SecondarySplitConfig,
    depth_map: Optional[np.ndarray] = None,
    depth_grad_map: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    if not os.path.exists(obj_path) or not os.path.exists(mask_path):
        return []

    points, colors = load_pcd_points_and_colors(obj_path)
    if points.shape[0] == 0:
        return []

    original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if original_mask is None:
        original_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    elif original_mask.shape[:2] != image_shape[:2]:
        original_mask = cv2.resize(
            original_mask,
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # 原始整体降噪
    points, colors = denoise_points_and_colors(
        points,
        colors,
        voxel_size=cfg.cluster_denoise_voxel_size,
        nb_neighbors=cfg.cluster_denoise_nb_neighbors,
        std_ratio=cfg.cluster_denoise_std_ratio,
        keep_largest_only=cfg.cluster_keep_largest_only,
    )

    if points.shape[0] == 0:
        return []

    # 如果不启用二次分割或不满足 split 条件
    if not cfg.enable_pointcloud_secondary_split or not is_candidate_for_secondary_split(points, cfg):
        refined_pcd_path = os.path.join(save_dir, f"{out_prefix}_refined.ply")
        refined_mask_path = os.path.join(save_dir, f"{out_prefix}_refined.png")

        save_pcd_points_and_colors(refined_pcd_path, points, colors)
        mask = project_points_to_mask(points, K, image_shape, dilate_ksize=5)
        cv2.imwrite(refined_mask_path, mask)

        return [{
            "pcd_path": refined_pcd_path,
            "mask_path": refined_mask_path,
            "debug": {
                "mode": "fallback_no_secondary_split",
                "num_points": int(points.shape[0]),
                "mask_area": int(np.count_nonzero(mask)),
            }
        }]

    if cfg.enable_recursive_split:
        split_results = recursive_split_pointcloud(
            points=points,
            colors=colors,
            K=K,
            image_shape=image_shape,
            cfg=cfg,
            depth_map=depth_map,
            depth_grad_map=depth_grad_map,
            original_mask=original_mask,
            depth=0,
        )
    else:
        split_results = [(points, colors)]

    # 重新对最终 split 结果做简单 debug 评估
    split_eval_debug = None
    if len(split_results) >= 2:
        eval_clusters = split_results[:2]
        _, _, split_eval_debug = evaluate_split_candidate(
            full_points=points,
            clusters=eval_clusters,
            K=K,
            image_shape=image_shape,
            cfg=cfg,
            depth_map=depth_map,
            depth_grad_map=depth_grad_map,
            original_mask=original_mask,
        )

    refined: List[Dict[str, Any]] = []
    for i, (sub_pts, sub_cols) in enumerate(split_results, start=1):
        sub_mask = project_points_to_mask(
            sub_pts,
            K,
            image_shape=image_shape,
            dilate_ksize=5,
        )

        area = int(np.count_nonzero(sub_mask > 0))
        if area < cfg.split_min_mask_area_px:
            continue

        conn2d = largest_component_ratio(sub_mask)
        if conn2d < cfg.min_largest_component_ratio:
            continue

        depth_conn = None
        if cfg.enable_depth_assisted_split and depth_map is not None:
            depth_conn = largest_depth_connected_component_ratio(
                mask=sub_mask,
                depth=depth_map,
                depth_thresh=cfg.depth_connect_thresh,
                min_valid_depth=cfg.min_valid_depth,
            )
            if depth_conn < cfg.min_depth_connected_ratio:
                continue

        if is_bad_shape_cluster(sub_pts, cfg):
            continue

        density_val = cluster_density_score(sub_pts)
        if density_val < cfg.min_cluster_density:
            continue

        if len(split_results) == 1:
            sub_pcd_path = os.path.join(save_dir, f"{out_prefix}_refined.ply")
            sub_mask_path = os.path.join(save_dir, f"{out_prefix}_refined.png")
        else:
            sub_pcd_path = os.path.join(save_dir, f"{out_prefix}_split_{i}.ply")
            sub_mask_path = os.path.join(save_dir, f"{out_prefix}_split_{i}.png")

        save_pcd_points_and_colors(sub_pcd_path, sub_pts, sub_cols)
        cv2.imwrite(sub_mask_path, sub_mask)

        refined.append({
            "pcd_path": sub_pcd_path,
            "mask_path": sub_mask_path,
            "debug": {
                "mode": "split" if len(split_results) > 1 else "refined_single",
                "sub_index": int(i),
                "num_points": int(sub_pts.shape[0]),
                "mask_area": int(area),
                "largest_component_ratio_2d": float(conn2d),
                "depth_connected_ratio": None if depth_conn is None else float(depth_conn),
                "density_score": float(density_val),
                "split_eval": split_eval_debug,
            }
        })

    if len(refined) > 0:
        return refined

    # 最终回退
    refined_pcd_path = os.path.join(save_dir, f"{out_prefix}_refined.ply")
    refined_mask_path = os.path.join(save_dir, f"{out_prefix}_refined.png")

    save_pcd_points_and_colors(refined_pcd_path, points, colors)
    mask = project_points_to_mask(points, K, image_shape, dilate_ksize=5)
    cv2.imwrite(refined_mask_path, mask)

    return [{
        "pcd_path": refined_pcd_path,
        "mask_path": refined_mask_path,
        "debug": {
            "mode": "final_fallback",
            "num_points": int(points.shape[0]),
            "mask_area": int(np.count_nonzero(mask)),
            "split_eval": split_eval_debug,
        }
    }]


# =========================================================
# Batch API (supports depth_map)
# =========================================================
def refine_objects_by_pointcloud_split(
    info: Dict[str, Any],
    K: np.ndarray,
    image_shape: Tuple[int, int],
    save_dir: str,
    cfg: SecondarySplitConfig,
    depth_map: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    批量处理多个 object point cloud，并支持深度图辅助判断。
    """
    os.makedirs(save_dir, exist_ok=True)

    depth_map_proc: Optional[np.ndarray] = None
    depth_grad_map: Optional[np.ndarray] = None

    if depth_map is not None:
        depth_map_proc = normalize_depth_map_if_needed(depth_map)
        if validate_depth_map_for_split(depth_map_proc, image_shape, allow_none=True):
            depth_map_proc = preprocess_depth_map(depth_map_proc, cfg)
            depth_grad_map = compute_depth_gradient_map(depth_map_proc, cfg)
        else:
            depth_map_proc = None
            depth_grad_map = None

    object_pcd_paths = info.get("object_pcd_paths", [])
    mask_paths = info.get("mask_paths", [])
    num_objs = min(len(object_pcd_paths), len(mask_paths))

    refined_pcd_paths: List[str] = []
    refined_mask_paths: List[str] = []
    split_debug: List[Dict[str, Any]] = []

    for i in range(num_objs):
        obj_path = object_pcd_paths[i]
        mask_path = mask_paths[i]

        refined_list = refine_one_object_by_pointcloud_split(
            obj_path=obj_path,
            mask_path=mask_path,
            K=K,
            image_shape=image_shape,
            save_dir=save_dir,
            out_prefix=f"object_{i+1}",
            cfg=cfg,
            depth_map=depth_map_proc,
            depth_grad_map=depth_grad_map,
        )

        for item in refined_list:
            refined_pcd_paths.append(item["pcd_path"])
            refined_mask_paths.append(item["mask_path"])

            dbg = {
                "source_object_index": i,
                "source_object_pcd": obj_path,
                "source_mask_path": mask_path,
                "refined_pcd_path": item["pcd_path"],
                "refined_mask_path": item["mask_path"],
            }

            if "debug" in item:
                dbg["debug"] = item["debug"]

            split_debug.append(dbg)

    info["object_pcd_paths"] = refined_pcd_paths
    info["mask_paths"] = refined_mask_paths
    info["saved_objects"] = len(refined_pcd_paths)
    info["saved_masks"] = len(refined_mask_paths)
    info["split_debug"] = split_debug

    return info


# =========================================================
# Compatibility wrapper
# =========================================================
def refine_objects_by_pointcloud_split_without_depth(
    info: Dict[str, Any],
    K: np.ndarray,
    image_shape: Tuple[int, int],
    save_dir: str,
    cfg: SecondarySplitConfig,
) -> Dict[str, Any]:
    """
    兼容旧调用方式：不使用 depth_map
    """
    return refine_objects_by_pointcloud_split(
        info=info,
        K=K,
        image_shape=image_shape,
        save_dir=save_dir,
        cfg=cfg,
        depth_map=None,
    )


# =========================================================
# Single-object debug API
# =========================================================
def debug_refine_single_object(
    obj_path: str,
    mask_path: str,
    K: np.ndarray,
    image_shape: Tuple[int, int],
    save_dir: str,
    cfg: SecondarySplitConfig,
    depth_map: Optional[np.ndarray] = None,
    out_prefix: str = "debug_object",
) -> Dict[str, Any]:
    """
    单对象调试接口，返回更详细的信息。
    """
    os.makedirs(save_dir, exist_ok=True)

    depth_map_proc = None
    if depth_map is not None:
        depth_map_proc = normalize_depth_map_if_needed(depth_map)
        if validate_depth_map_for_split(depth_map_proc, image_shape, allow_none=True):
            depth_map_proc = preprocess_depth_map(depth_map_proc, cfg)
        else:
            depth_map_proc = None

    refined_list = refine_one_object_by_pointcloud_split(
        obj_path=obj_path,
        mask_path=mask_path,
        K=K,
        image_shape=image_shape,
        save_dir=save_dir,
        out_prefix=out_prefix,
        cfg=cfg,
        depth_map=depth_map_proc,
    )

    result = {
        "input_obj_path": obj_path,
        "input_mask_path": mask_path,
        "num_outputs": len(refined_list),
        "outputs": refined_list,
    }
    return result


# =========================================================
# Validation helpers
# =========================================================
def validate_depth_map_for_split(
    depth_map: Optional[np.ndarray],
    image_shape: Tuple[int, int],
    allow_none: bool = True,
) -> bool:
    """
    检查 depth_map 是否可用于 split 辅助判断。
    """
    if depth_map is None:
        return bool(allow_none)

    if not isinstance(depth_map, np.ndarray):
        return False

    if depth_map.ndim != 2:
        return False

    h, w = image_shape[:2]
    if depth_map.shape[0] != h or depth_map.shape[1] != w:
        return False

    return True


def normalize_depth_map_if_needed(depth_map: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    可选的简单归一化/类型修正：
    - 若是 uint16，通常默认按毫米转米
    - 若已是 float，则直接转 float32
    """
    if depth_map is None:
        return None

    if depth_map.dtype == np.uint16:
        return depth_map.astype(np.float32) / 1000.0

    return depth_map.astype(np.float32)

