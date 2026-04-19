from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

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

    # ---------- secondary split trigger ----------
    # 只有足够大的候选物体才允许进入二次分割
    secondary_split_min_points: int = 2500
    secondary_split_min_extent_max: float = 0.12

    # ---------- DBSCAN on single merged object ----------
    secondary_split_voxel_size: float = 0.004
    secondary_split_dbscan_eps: float = 0.015
    secondary_split_dbscan_min_points: int = 100

    # ---------- split acceptance ----------
    secondary_split_min_cluster_ratio: float = 0.25
    secondary_split_min_subcluster_points: int = 400
    secondary_split_min_center_dist: float = 0.05
    secondary_split_min_gap_dist: float = 0.015
    secondary_split_min_mask_area_px: int = 250

    # ---------- quality filter ----------
    reject_slender_ratio_1: float = 8.0
    reject_slender_ratio_2: float = 6.0
    min_largest_component_ratio: float = 0.85
    min_cluster_density: float = 12000.0

    # split 接受时要求比分前明显更合理
    split_score_margin: float = 20.0


# =========================================================
# IO / basic point cloud utils
# =========================================================
def load_pcd_points_and_colors(path: str) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray]:
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
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return points, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and colors.shape[0] == points.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    # 1) voxel
    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    # 2) statistical outlier removal
    try:
        if len(pcd.points) >= nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
    except Exception:
        pass

    # 3) keep largest cluster if needed
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
        return 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    total = int(np.sum(areas))
    if total <= 0:
        return 0.0
    return float(np.max(areas)) / float(total)


def project_points_to_mask(
    points_c: np.ndarray,
    K: np.ndarray,
    image_shape: tuple[int, int],
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


def cluster_quality_score(points: np.ndarray, mask: np.ndarray, cfg: SecondarySplitConfig) -> float:
    if points.shape[0] < 50:
        return -1e6

    if is_bad_shape_cluster(points, cfg):
        return -1000.0

    density = cluster_density_score(points)
    if density < cfg.min_cluster_density:
        return -500.0

    conn_ratio = largest_component_ratio(mask)
    if conn_ratio < cfg.min_largest_component_ratio:
        return -300.0

    return 0.01 * float(points.shape[0]) + 0.0001 * density + 2.0 * conn_ratio


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
# Secondary split logic
# =========================================================
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


def split_pointcloud_dbscan(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
    eps: float,
    min_points: int,
    cfg: SecondarySplitConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if points.shape[0] == 0:
        return []

    pts_ds, cols_ds = voxel_downsample_points_and_colors(points, colors, voxel_size)
    if pts_ds.shape[0] == 0:
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_ds.astype(np.float64))
    if cols_ds.shape[0] == pts_ds.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(cols_ds.astype(np.float64))

    labels = np.array(
        pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False
        )
    )

    if labels.size == 0 or np.all(labels < 0):
        return []

    valid = labels[labels >= 0]
    if valid.size == 0:
        return []

    counts = np.bincount(valid)
    order = np.argsort(counts)[::-1]

    clusters = []
    for lab in order:
        idx = np.where(labels == lab)[0]
        sub = pcd.select_by_index(idx)

        sub_pts = np.asarray(sub.points).astype(np.float32)
        sub_cols = np.asarray(sub.colors).astype(np.float32)
        if sub_cols.shape[0] != sub_pts.shape[0]:
            sub_cols = np.zeros((sub_pts.shape[0], 3), dtype=np.float32)

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


def evaluate_split_candidate(
    full_points: np.ndarray,
    clusters: list[tuple[np.ndarray, np.ndarray]],
    K: np.ndarray,
    image_shape: tuple[int, int],
    cfg: SecondarySplitConfig,
) -> tuple[bool, float]:
    if len(clusters) < 2:
        return False, -1e9

    total_n = full_points.shape[0]
    if total_n < cfg.secondary_split_min_points:
        return False, -1e9

    sizes = [c[0].shape[0] for c in clusters]
    n1, n2 = sizes[0], sizes[1]

    if n1 < cfg.secondary_split_min_subcluster_points or n2 < cfg.secondary_split_min_subcluster_points:
        return False, -1e9

    if n2 < total_n * cfg.secondary_split_min_cluster_ratio:
        return False, -1e9

    c1 = np.median(clusters[0][0], axis=0)
    c2 = np.median(clusters[1][0], axis=0)
    center_dist = float(np.linalg.norm(c1 - c2))
    if center_dist < cfg.secondary_split_min_center_dist:
        return False, -1e9

    gap_dist = min_intercluster_distance(clusters[0][0], clusters[1][0])
    if gap_dist < cfg.secondary_split_min_gap_dist:
        return False, -1e9

    full_mask = project_points_to_mask(full_points, K, image_shape, dilate_ksize=5)
    score_before = cluster_quality_score(full_points, full_mask, cfg)

    score_after = 0.0
    for pts, _ in clusters[:2]:
        sub_mask = project_points_to_mask(pts, K, image_shape, dilate_ksize=5)
        score_after += cluster_quality_score(pts, sub_mask, cfg)

    accept = score_after > score_before + cfg.split_score_margin
    return accept, score_after - score_before


def refine_one_object_by_pointcloud_split(
    obj_path: str,
    mask_path: str,
    K: np.ndarray,
    image_shape: tuple[int, int],
    save_dir: str,
    out_prefix: str,
    cfg: SecondarySplitConfig,
) -> list[dict]:
    if not os.path.exists(obj_path) or not os.path.exists(mask_path):
        return []

    points, colors = load_pcd_points_and_colors(obj_path)
    if points.shape[0] == 0:
        return []

    # 先对原始物体整体降噪一次
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

    # 不值得二次分割时，直接输出 refined
    if not cfg.enable_pointcloud_secondary_split or not is_candidate_for_secondary_split(points, cfg):
        refined_pcd_path = os.path.join(save_dir, f"{out_prefix}_refined.ply")
        refined_mask_path = os.path.join(save_dir, f"{out_prefix}_refined.png")

        save_pcd_points_and_colors(refined_pcd_path, points, colors)
        mask = project_points_to_mask(points, K, image_shape, dilate_ksize=5)
        cv2.imwrite(refined_mask_path, mask)

        return [{
            "pcd_path": refined_pcd_path,
            "mask_path": refined_mask_path,
        }]

    # 只保留 DBSCAN 候选，禁用主轴硬切
    clusters_db = split_pointcloud_dbscan(
        points, colors,
        voxel_size=cfg.secondary_split_voxel_size,
        eps=cfg.secondary_split_dbscan_eps,
        min_points=cfg.secondary_split_dbscan_min_points,
        cfg=cfg,
    )

    candidates = []
    ok_db, gain_db = evaluate_split_candidate(points, clusters_db, K, image_shape, cfg)
    if ok_db:
        candidates.append(("dbscan", gain_db, clusters_db))

    if len(candidates) == 0:
        refined_pcd_path = os.path.join(save_dir, f"{out_prefix}_refined.ply")
        refined_mask_path = os.path.join(save_dir, f"{out_prefix}_refined.png")

        save_pcd_points_and_colors(refined_pcd_path, points, colors)
        mask = project_points_to_mask(points, K, image_shape, dilate_ksize=5)
        cv2.imwrite(refined_mask_path, mask)

        return [{
            "pcd_path": refined_pcd_path,
            "mask_path": refined_mask_path,
        }]

    candidates.sort(key=lambda x: x[1], reverse=True)
    _, _, best_clusters = candidates[0]

    refined = []
    for i, (sub_pts, sub_cols) in enumerate(best_clusters[:2], start=1):
        sub_mask = project_points_to_mask(
            sub_pts,
            K,
            image_shape=image_shape,
            dilate_ksize=5,
        )

        area = int(np.count_nonzero(sub_mask > 0))
        if area < cfg.secondary_split_min_mask_area_px:
            continue

        if largest_component_ratio(sub_mask) < cfg.min_largest_component_ratio:
            continue

        if is_bad_shape_cluster(sub_pts, cfg):
            continue

        if cluster_density_score(sub_pts) < cfg.min_cluster_density:
            continue

        sub_pcd_path = os.path.join(save_dir, f"{out_prefix}_split_{i}.ply")
        sub_mask_path = os.path.join(save_dir, f"{out_prefix}_split_{i}.png")

        save_pcd_points_and_colors(sub_pcd_path, sub_pts, sub_cols)
        cv2.imwrite(sub_mask_path, sub_mask)

        refined.append({
            "pcd_path": sub_pcd_path,
            "mask_path": sub_mask_path,
        })

    if len(refined) >= 2:
        return refined

    # 若过滤后不够两个，回退
    refined_pcd_path = os.path.join(save_dir, f"{out_prefix}_refined.ply")
    refined_mask_path = os.path.join(save_dir, f"{out_prefix}_refined.png")

    save_pcd_points_and_colors(refined_pcd_path, points, colors)
    mask = project_points_to_mask(points, K, image_shape, dilate_ksize=5)
    cv2.imwrite(refined_mask_path, mask)

    return [{
        "pcd_path": refined_pcd_path,
        "mask_path": refined_mask_path,
    }]


def refine_objects_by_pointcloud_split(
    info: Dict[str, Any],
    K: np.ndarray,
    image_shape: tuple[int, int],
    save_dir: str,
    cfg: SecondarySplitConfig,
) -> Dict[str, Any]:
    object_pcd_paths = info.get("object_pcd_paths", [])
    mask_paths = info.get("mask_paths", [])
    num_objs = min(len(object_pcd_paths), len(mask_paths))

    refined_pcd_paths = []
    refined_mask_paths = []

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
        )

        for item in refined_list:
            refined_pcd_paths.append(item["pcd_path"])
            refined_mask_paths.append(item["mask_path"])

    info["object_pcd_paths"] = refined_pcd_paths
    info["mask_paths"] = refined_mask_paths
    info["saved_objects"] = len(refined_pcd_paths)
    info["saved_masks"] = len(refined_mask_paths)

    return info