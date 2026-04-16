from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import os

import numpy as np
import cv2
import open3d as o3d

from table_segmentation import TableSegConfig, fit_plane_from_pcd, _normalize


@dataclass
class MaskGenConfig:
    # ---------- table ----------
    table_band_thickness: float = 0.012
    table_close_ksize: int = 21
    table_dilate_ksize: int = 3
    table_erode_for_objects_ksize: int = 21
    table_min_area_px: int = 5000

    # ---------- tabletop objects ----------
    object_plane_clearance: float = 0.015
    object_max_distance: float = 0.60
    object_open_ksize: int = 3
    object_close_ksize: int = 3
    object_dilate_ksize: int = 1
    object_min_area_px: int = 400

    # ---------- clustering ----------
    dbscan_eps: float = 0.015
    dbscan_min_points: int = 80
    min_cluster_points: int = 200

    # ---------- single object mask ----------
    object_mask_close_ksize: int = 3
    object_mask_dilate_ksize: int = 1
    min_object_mask_area: int = 200

    # ---------- merged-cluster split ----------
    enable_split_merged_cluster: bool = True

    # 粗 cluster 多大时，认为值得检查是否连体
    split_check_min_points: int = 1200
    split_check_min_mask_area: int = 2500
    split_check_bbox_max_side_px: int = 120

    # 高度图 watershed
    split_gaussian_ksize: int = 5
    split_dist_thresh_ratio: float = 0.38
    split_height_percentile_clip_low: float = 5.0
    split_height_percentile_clip_high: float = 99.0
    split_min_peak_components: int = 2
    split_min_submask_area: int = 300
    split_min_subcluster_points: int = 150

    # 是否把二次拆分结果保存调试图
    save_split_debug: bool = True

    # ---------- single-frame tabletop support filter ----------
    support_check_bottom_band: float = 0.02
    support_check_min_bottom_pixels: int = 80
    support_check_min_inside_ratio: float = 0.60

    # 距图像边缘太近的实例更容易是误检
    edge_margin_px: int = 12
    edge_reject_if_bottom_inside_ratio_below: float = 0.75
    # ---------- output ----------
    save_dir: str = "output"
    table_pcd_name: str = "table.ply"
    object_pcd_prefix: str = "object_"
    mask_prefix: str = "obj_mask_"


# =========================================================
# Basic utils
# =========================================================
def save_pointcloud(path: str, pcd: o3d.geometry.PointCloud):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    o3d.io.write_point_cloud(path, pcd)


def save_plane_model(path: str, n: np.ndarray, d: float):
    data = {"n": np.asarray(n, dtype=np.float64), "d": float(d)}
    np.save(path, data, allow_pickle=True)


def make_pointcloud_from_rgbd(
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale: float,
    K: np.ndarray
) -> o3d.geometry.PointCloud:
    depth = depth_u16.astype(np.float32) * float(depth_scale)

    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    valid = z > 1e-6

    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy

    pts = np.stack([x, y, z], axis=-1)[valid]
    colors = color_bgr[:, :, ::-1].astype(np.float32) / 255.0
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def build_xyz_map_from_depth(depth_u16: np.ndarray, depth_scale: float, K: np.ndarray) -> np.ndarray:
    depth = depth_u16.astype(np.float32) * float(depth_scale)

    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy

    xyz = np.stack([x, y, z], axis=-1)
    return xyz

def orient_plane_normal_for_tabletop_objects(
    xyz: np.ndarray,
    table_mask: np.ndarray,
    n: np.ndarray,
    d: float,
    clearance: float,
    max_distance: float,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    在 (n, d) 和 (-n, -d) 两种平面方向中，选择更可能让“桌上物体”落在正侧的方向。

    判据：
    仅在桌面区域内统计，比较两种方向下满足
        clearance <= signed_dist <= max_distance
    的候选像素数量，谁更多就选谁。
    """
    z = xyz[..., 2]
    valid_depth = z > 1e-6
    table_region = table_mask > 0

    n1 = _normalize(n)
    d1 = float(d)

    s1 = xyz @ n1.reshape(3, 1)
    s1 = s1[..., 0] + d1
    cand1 = (
        valid_depth &
        table_region &
        (s1 >= clearance) &
        (s1 <= max_distance)
    )
    cnt1 = int(np.sum(cand1))

    n2 = -n1
    d2 = -d1

    s2 = xyz @ n2.reshape(3, 1)
    s2 = s2[..., 0] + d2
    cand2 = (
        valid_depth &
        table_region &
        (s2 >= clearance) &
        (s2 <= max_distance)
    )
    cnt2 = int(np.sum(cand2))

    if cnt2 > cnt1:
        return n2, d2, {
            "plane_flipped": True,
            "candidate_count_original": cnt1,
            "candidate_count_flipped": cnt2,
        }

    return n1, d1, {
        "plane_flipped": False,
        "candidate_count_original": cnt1,
        "candidate_count_flipped": cnt2,
    }

def make_pcd_from_xyz_and_mask(
    color_bgr: np.ndarray,
    xyz: np.ndarray,
    pixel_mask: np.ndarray,
) -> o3d.geometry.PointCloud:
    valid = (pixel_mask > 0) & (xyz[..., 2] > 1e-6)
    pts = xyz[valid]
    colors = color_bgr[:, :, ::-1].astype(np.float32) / 255.0
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    if pts.shape[0] == 0:
        return pcd

    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd

def make_pcd_from_xyz_mask_and_reference_mask(
    color_bgr: np.ndarray,
    xyz: np.ndarray,
    submask: np.ndarray,
    reference_mask: np.ndarray,
) -> o3d.geometry.PointCloud:
    valid = (submask > 0) & (reference_mask > 0) & (xyz[..., 2] > 1e-6)
    pts = xyz[valid]
    colors = color_bgr[:, :, ::-1].astype(np.float32) / 255.0
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    if pts.shape[0] == 0:
        return pcd

    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd

def project_points_to_mask(pts_cam: np.ndarray, K: np.ndarray, H: int, W: int) -> np.ndarray:
    if pts_cam.shape[0] == 0:
        return np.zeros((H, W), dtype=np.uint8)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    z = pts_cam[:, 2]
    valid = z > 1e-6
    pts = pts_cam[valid]
    if pts.shape[0] == 0:
        return np.zeros((H, W), dtype=np.uint8)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    u = (fx * (x / z) + cx).astype(np.int32)
    v = (fy * (y / z) + cy).astype(np.int32)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[inside]
    v = v[inside]

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[v, u] = 255
    return mask


# =========================================================
# Mask helpers
# =========================================================
def keep_largest_component(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    
    if num_labels <= 1:
        out = (m * 255).astype(np.uint8)
        if int((out > 0).sum()) < min_area:
            return np.zeros_like(out)
        return out

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    out = ((labels == largest).astype(np.uint8) * 255)

    if int((out > 0).sum()) < min_area:
        return np.zeros_like(out)
    return out


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    
    out = np.zeros_like(m, dtype=np.uint8)
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == lab] = 255
    return out

def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1, y1


def connected_component_count(mask: np.ndarray, min_area: int = 1) -> int:
    m = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    cnt = 0
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area:
            cnt += 1
    return cnt

# =========================================================
# Table extraction
# =========================================================
def extract_plane_band_points(
    pcd: o3d.geometry.PointCloud,
    n: np.ndarray,
    d: float,
    thickness: float
) -> o3d.geometry.PointCloud:
    n = _normalize(n)
    pts = np.asarray(pcd.points)
    s = pts @ n + float(d)
    idx = np.where(np.abs(s) <= thickness)[0]
    return pcd.select_by_index(idx)

def build_table_mask(
    raw_table_pcd: o3d.geometry.PointCloud,
    K: np.ndarray,
    H: int,
    W: int,
    cfg: MaskGenConfig
) -> np.ndarray:
    pts = np.asarray(raw_table_pcd.points)
    mask = project_points_to_mask(pts, K, H, W)

    if cfg.table_close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.table_close_ksize, cfg.table_close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if cfg.table_dilate_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.table_dilate_ksize, cfg.table_dilate_ksize))
        mask = cv2.dilate(mask, k, iterations=1)

    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    mask = cv2.bitwise_or(mask, flood_inv)

    mask = keep_largest_component(mask, min_area=cfg.table_min_area_px)
    return mask


def build_table_only_mask(
    xyz: np.ndarray,
    table_mask: np.ndarray,
    n: np.ndarray,
    d: float,
    cfg: MaskGenConfig
) -> np.ndarray:
    z = xyz[..., 2]
    valid_depth = z > 1e-6

    n = _normalize(n)
    signed_dist = xyz @ n.reshape(3, 1)
    signed_dist = signed_dist[..., 0] + float(d)

    keep = (
        valid_depth &
        (table_mask > 0) &
        (np.abs(signed_dist) <= cfg.table_band_thickness)
    )
    return (keep.astype(np.uint8) * 255)


# =========================================================
# Tabletop objects
# =========================================================
def build_tabletop_object_pixels_mask(
    xyz: np.ndarray,
    table_mask: np.ndarray,
    n: np.ndarray,
    d: float,
    cfg: MaskGenConfig,
) -> np.ndarray:
    z = xyz[..., 2]
    valid_depth = z > 1e-6

    table_inner = table_mask.copy()
    if cfg.table_erode_for_objects_ksize > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (cfg.table_erode_for_objects_ksize, cfg.table_erode_for_objects_ksize)
        )
        table_inner = cv2.erode(table_inner, k, iterations=1)

    n = _normalize(n)
    signed_dist = xyz @ n.reshape(3, 1)
    signed_dist = signed_dist[..., 0] + float(d)

    object_region = (
        valid_depth &
        (table_inner > 0) &
        (signed_dist >= cfg.object_plane_clearance) &
        (signed_dist <= cfg.object_max_distance)
    )

    mask = (object_region.astype(np.uint8) * 255)

    if cfg.object_open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.object_open_ksize, cfg.object_open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if cfg.object_close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.object_close_ksize, cfg.object_close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if cfg.object_dilate_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.object_dilate_ksize, cfg.object_dilate_ksize))
        mask = cv2.dilate(mask, k, iterations=1)

    mask = remove_small_components(mask, min_area=cfg.object_min_area_px)
    return mask


# =========================================================
# Clustering
# =========================================================
def cluster_objects_3d(pcd_obj: o3d.geometry.PointCloud, cfg: MaskGenConfig) -> List[np.ndarray]:
    if len(pcd_obj.points) == 0:
        return []

    labels = np.array(
        pcd_obj.cluster_dbscan(
            eps=cfg.dbscan_eps,
            min_points=cfg.dbscan_min_points,
            print_progress=False
        ),
        dtype=int
    )
    if labels.size == 0:
        return []

    clusters = []
    for lab in range(labels.max() + 1):
        idx = np.where(labels == lab)[0]
        if idx.size >= cfg.min_cluster_points:
            clusters.append(idx)

    clusters.sort(key=lambda x: x.size, reverse=True)
    return clusters


def cluster_extent_ok(obj_pcd: o3d.geometry.PointCloud, n: np.ndarray, d: float) -> Tuple[bool, Dict[str, float]]:
    pts = np.asarray(obj_pcd.points)
    if pts.shape[0] == 0:
        return False, {"reason": -1.0}

    bbox = obj_pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    max_extent = float(extent.max())

    n = _normalize(n)
    signed_h = pts @ n + float(d)
    height_extent = float(signed_h.max() - signed_h.min())
    center_height = float(np.median(signed_h))

    ok = True
    if max_extent < 0.01:
        ok = False
    if max_extent > 0.50:
        ok = False
    if height_extent > 0.50:
        ok = False
    if center_height < 0.01:
        ok = False

    return ok, {
        "max_extent": max_extent,
        "height_extent": height_extent,
        "center_height": center_height,
    }


# =========================================================
# Single object mask
# =========================================================
def postprocess_single_object_mask(mask: np.ndarray, cfg: MaskGenConfig) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255

    if cfg.object_mask_close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.object_mask_close_ksize, cfg.object_mask_close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    h, w = m.shape
    flood = m.copy()
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    m = cv2.bitwise_or(m, flood_inv)

    if cfg.object_mask_dilate_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.object_mask_dilate_ksize, cfg.object_mask_dilate_ksize))
        m = cv2.dilate(m, k, iterations=1)

    return m

def filter_mask_by_single_frame_table_support(
    sub_mask: np.ndarray,
    xyz: np.ndarray,
    table_mask: np.ndarray,
    n: np.ndarray,
    d: float,
    cfg: MaskGenConfig,
) -> Tuple[bool, Dict[str, Any]]:
    """
    单帧下判断该实例是否真的像是“摆在桌上”
    """
    valid = (sub_mask > 0) & (xyz[..., 2] > 1e-6)
    result = {
        "keep": False,
        "bottom_inside_ratio": 0.0,
        "num_bottom_pixels": 0,
        "touches_image_edge": False,
        "reject_reason": "",
    }

    if not np.any(valid):
        result["reject_reason"] = "empty_mask_or_invalid_depth"
        return False, result

    n = _normalize(n)
    signed_h = xyz @ n.reshape(3, 1)
    signed_h = signed_h[..., 0] + float(d)

    vals = signed_h[valid]
    if vals.size == 0:
        result["reject_reason"] = "no_signed_height_values"
        return False, result

    h_bottom = float(np.percentile(vals, 5.0))
    bottom_region = valid & (signed_h <= (h_bottom + cfg.support_check_bottom_band))

    num_bottom = int(np.sum(bottom_region))
    result["num_bottom_pixels"] = num_bottom

    if num_bottom < cfg.support_check_min_bottom_pixels:
        result["reject_reason"] = "too_few_bottom_pixels"
        return False, result

    # 底部是否落在桌面内部
    table_inner = table_mask > 0
    inside = bottom_region & table_inner
    inside_ratio = float(np.sum(inside)) / float(num_bottom + 1e-6)
    result["bottom_inside_ratio"] = inside_ratio

    # 是否贴近图像边缘
    bbox = bbox_from_mask(sub_mask)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        H, W = sub_mask.shape
        touches_edge = (
            x0 < cfg.edge_margin_px or
            y0 < cfg.edge_margin_px or
            x1 >= W - cfg.edge_margin_px or
            y1 >= H - cfg.edge_margin_px
        )
    else:
        touches_edge = False

    result["touches_image_edge"] = touches_edge

    if inside_ratio < cfg.support_check_min_inside_ratio:
        result["reject_reason"] = "bottom_inside_ratio_too_low"
        return False, result

    if touches_edge and inside_ratio < cfg.edge_reject_if_bottom_inside_ratio_below:
        result["reject_reason"] = "edge_instance_with_low_support"
        return False, result

    result["keep"] = True
    return True, result

# =========================================================
# Split merged cluster
# =========================================================
def is_merged_cluster_candidate(
    obj_pcd: o3d.geometry.PointCloud,
    mask_i: np.ndarray,
    cfg: MaskGenConfig,
) -> Tuple[bool, Dict[str, Any]]:
    pts = np.asarray(obj_pcd.points)
    num_points = int(pts.shape[0])
    mask_area = int((mask_i > 0).sum())

    bbox = bbox_from_mask(mask_i)
    bbox_w = 0
    bbox_h = 0
    bbox_max_side = 0
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        bbox_w = int(x1 - x0 + 1)
        bbox_h = int(y1 - y0 + 1)
        bbox_max_side = max(bbox_w, bbox_h)

    suspicious = (
        (num_points >= cfg.split_check_min_points) or
        (mask_area >= cfg.split_check_min_mask_area) or
        (bbox_max_side >= cfg.split_check_bbox_max_side_px)
    )

    return suspicious, {
        "num_points": num_points,
        "mask_area": mask_area,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "bbox_max_side": bbox_max_side,
    }


def build_height_map_in_mask(
    xyz: np.ndarray,
    instance_mask: np.ndarray,
    n: np.ndarray,
    d: float,
    cfg: MaskGenConfig,
) -> np.ndarray:
    n = _normalize(n)
    signed_h = xyz @ n.reshape(3, 1)
    signed_h = signed_h[..., 0] + float(d)

    valid = (instance_mask > 0) & (xyz[..., 2] > 1e-6)
    if not np.any(valid):
        return np.zeros(instance_mask.shape, dtype=np.uint8)

    vals = signed_h[valid].astype(np.float32)
    low = np.percentile(vals, cfg.split_height_percentile_clip_low)
    high = np.percentile(vals, cfg.split_height_percentile_clip_high)

    if high <= low + 1e-6:
        out = np.zeros(instance_mask.shape, dtype=np.uint8)
        out[valid] = 255
        return out

    vals_clip = np.clip(signed_h, low, high)
    vals_norm = (vals_clip - low) / (high - low + 1e-6)

    height_u8 = np.zeros(instance_mask.shape, dtype=np.uint8)
    height_u8[valid] = np.clip(vals_norm[valid] * 255.0, 0, 255).astype(np.uint8)

    k = cfg.split_gaussian_ksize
    if k >= 3 and k % 2 == 1:
        height_u8 = cv2.GaussianBlur(height_u8, (k, k), 0)

    return height_u8


def save_split_debug_images(
    save_dir: str,
    cluster_id: int,
    mask_i: np.ndarray,
    height_u8: np.ndarray,
    sure_fg: np.ndarray,
    markers_ws: np.ndarray,
):
    os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(os.path.join(save_dir, f"cluster_{cluster_id:02d}_mask.png"), mask_i)
    cv2.imwrite(os.path.join(save_dir, f"cluster_{cluster_id:02d}_height.png"), height_u8)
    cv2.imwrite(os.path.join(save_dir, f"cluster_{cluster_id:02d}_sure_fg.png"), sure_fg)

    if markers_ws.size > 0:
        vis = np.zeros((markers_ws.shape[0], markers_ws.shape[1], 3), dtype=np.uint8)
        uniq = np.unique(markers_ws)
        rng = np.random.default_rng(42)
        color_map = {}
        for lab in uniq:
            if lab <= 1:
                continue
            color_map[int(lab)] = rng.integers(0, 255, size=3, dtype=np.uint8)

        for lab, c in color_map.items():
            vis[markers_ws == lab] = c

        vis[markers_ws == -1] = (0, 0, 255)
        cv2.imwrite(os.path.join(save_dir, f"cluster_{cluster_id:02d}_watershed.png"), vis)


def split_cluster_by_height_watershed(
    cluster_id: int,
    color_bgr: np.ndarray,
    xyz: np.ndarray,
    mask_i: np.ndarray,
    n: np.ndarray,
    d: float,
    cfg: MaskGenConfig,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    输入一个粗 cluster 的 2D mask，在该 mask 内构建相对桌面的高度图，
    用 distance transform + watershed 做二次拆分。
    """
    binary = ((mask_i > 0).astype(np.uint8) * 255)
    area = int((binary > 0).sum())

    if area < cfg.split_min_submask_area * 2:
        return [mask_i], {
            "split_applied": False,
            "split_reason": "mask_too_small",
            "num_submasks": 1,
        }

    # 1) 高度图（主要用于调试和后续可扩展）
    height_u8 = build_height_map_in_mask(
        xyz=xyz,
        instance_mask=binary,
        n=n,
        d=d,
        cfg=cfg,
    )

    # 2) 二值 mask 上做距离变换
    fg = (binary > 0).astype(np.uint8)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)

    if float(dist.max()) < 1.0:
        return [mask_i], {
            "split_applied": False,
            "split_reason": "distance_transform_too_small",
            "num_submasks": 1,
        }

    # 3) 前景种子
    thresh = cfg.split_dist_thresh_ratio * float(dist.max())
    sure_fg = (dist > thresh).astype(np.uint8) * 255
    sure_fg = remove_small_components(sure_fg, min_area=cfg.split_min_submask_area)

    peak_components = connected_component_count(sure_fg, min_area=cfg.split_min_submask_area)
    if peak_components < cfg.split_min_peak_components:
        return [mask_i], {
            "split_applied": False,
            "split_reason": f"peak_components<{cfg.split_min_peak_components}",
            "num_submasks": 1,
        }

    # 4) watershed
    unknown = cv2.subtract(binary, sure_fg)
    _, markers = cv2.connectedComponents((sure_fg > 0).astype(np.uint8))
    markers = markers + 1
    markers[unknown > 0] = 0

    ws_img = color_bgr.copy()
    markers_ws = cv2.watershed(ws_img, markers.astype(np.int32))

    labels = np.unique(markers_ws)
    sub_masks: List[np.ndarray] = []

    for lab in labels:
        if lab <= 1:
            continue

        sub = np.zeros_like(binary)
        sub[markers_ws == lab] = 255

        # 保证仍在原始 cluster 区域内
        sub = cv2.bitwise_and(sub, binary)
        sub = remove_small_components(sub, min_area=cfg.split_min_submask_area)

        if int((sub > 0).sum()) >= cfg.split_min_submask_area:
            sub_masks.append(sub)

    if len(sub_masks) <= 1:
        return [mask_i], {
            "split_applied": False,
            "split_reason": "watershed_not_split",
            "num_submasks": 1,
        }

    if cfg.save_split_debug:
        debug_dir = os.path.join(cfg.save_dir, "split_debug")
        save_split_debug_images(
            save_dir=debug_dir,
            cluster_id=cluster_id,
            mask_i=binary,
            height_u8=height_u8,
            sure_fg=sure_fg,
            markers_ws=markers_ws,
        )

    return sub_masks, {
        "split_applied": True,
        "split_reason": "watershed_split",
        "num_submasks": len(sub_masks),
        "peak_components": int(peak_components),
        "dist_max": float(dist.max()),
        "dist_thresh": float(thresh),
    }

# =========================================================
# Main
# =========================================================
def first_frame_get_and_save_masks(
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale: float,
    K: np.ndarray,
    make_pointcloud_fn,
    table_cfg: Optional[TableSegConfig] = None,
    mask_cfg: Optional[MaskGenConfig] = None,
    save_plane_path: str = "table_plane.npy",
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    table_cfg = table_cfg or TableSegConfig()
    mask_cfg = mask_cfg or MaskGenConfig()

    H, W = color_bgr.shape[:2]
    os.makedirs(mask_cfg.save_dir, exist_ok=True)

    # 1) 全场景点云
    pcd_cam: o3d.geometry.PointCloud = make_pointcloud_fn(
        color_bgr, depth_u16, depth_scale, K
    )

    # 2) 拟合桌面平面
    n, d, info_seg = fit_plane_from_pcd(pcd_cam, table_cfg)
    if n is None:
        return [], {"status": "fail_table_seg", **info_seg}

    xyz = build_xyz_map_from_depth(depth_u16, depth_scale, K)

    # 3) 先用当前方向生成桌面 mask
    raw_table_pcd = extract_plane_band_points(
        pcd_cam,
        n,
        d,
        thickness=mask_cfg.table_band_thickness
    )
    table_mask = build_table_mask(raw_table_pcd, K, H, W, mask_cfg)

    if int((table_mask > 0).sum()) < mask_cfg.table_min_area_px:
        return [], {
            "status": "fail_table_component",
            "plane_path": save_plane_path,
            **info_seg,
        }

    # 4) 自动统一平面法向方向
    n, d, plane_orient_info = orient_plane_normal_for_tabletop_objects(
        xyz=xyz,
        table_mask=table_mask,
        n=n,
        d=d,
        clearance=mask_cfg.object_plane_clearance,
        max_distance=mask_cfg.object_max_distance,
    )

    # 重新保存翻正后的平面
    save_plane_model(save_plane_path, n, d)

    if int((table_mask > 0).sum()) < mask_cfg.table_min_area_px:
        return [], {
            "status": "fail_table_component",
            "plane_path": save_plane_path,
            **info_seg,
        }

    # 4) 纯桌面点云（不含桌上物体）
    table_only_mask = build_table_only_mask(
        xyz=xyz,
        table_mask=table_mask,
        n=n,
        d=d,
        cfg=mask_cfg,
    )
    table_pcd = make_pcd_from_xyz_and_mask(color_bgr, xyz, table_only_mask)
    table_pcd_path = os.path.join(mask_cfg.save_dir, mask_cfg.table_pcd_name)
    save_pointcloud(table_pcd_path, table_pcd)

    # 5) 桌面上的物体像素
    object_pixels_mask = build_tabletop_object_pixels_mask(
        xyz=xyz,
        table_mask=table_mask,
        n=n,
        d=d,
        cfg=mask_cfg,
    )

    if int((object_pixels_mask > 0).sum()) < mask_cfg.object_min_area_px:
        return [], {
            "status": "no_object_pixels_on_table",
            "table_pcd_path": table_pcd_path,
            "plane_path": save_plane_path,
            **info_seg,
        }

    # 6) 桌面上的物体总点云
    pcd_obj_all = make_pcd_from_xyz_and_mask(color_bgr, xyz, object_pixels_mask)
    if len(pcd_obj_all.points) == 0:
        return [], {
            "status": "objects_pcd_empty",
            "table_pcd_path": table_pcd_path,
            "plane_path": save_plane_path,
            **info_seg,
        }

    # 7) 粗聚类
    clusters = cluster_objects_3d(pcd_obj_all, mask_cfg)
    if len(clusters) == 0:
        return [], {
            "status": "no_clusters",
            "table_pcd_path": table_pcd_path,
            "plane_path": save_plane_path,
            **info_seg,
        }

    pts_all = np.asarray(pcd_obj_all.points)
    masks: List[np.ndarray] = []
    object_pcd_paths: List[str] = []
    mask_paths: List[str] = []
    cluster_debug: List[Dict[str, Any]] = []

    saved_objects = 0
    saved_masks = 0

    for i, idx in enumerate(clusters, start=1):
        obj_i_pcd = pcd_obj_all.select_by_index(idx)

        ok_cluster, extent_info = cluster_extent_ok(obj_i_pcd, n, d)
        item = {
            "cluster_id": i,
            "num_points": int(len(obj_i_pcd.points)),
            **extent_info,
            "accepted": bool(ok_cluster),
        }
        if not ok_cluster:
            cluster_debug.append(item)
            continue

        pts_i = pts_all[idx]
        mask_i = project_points_to_mask(pts_i, K, H, W)
        mask_i = postprocess_single_object_mask(mask_i, mask_cfg)

        initial_area = int((mask_i > 0).sum())
        item["initial_mask_area"] = initial_area

        if initial_area < mask_cfg.min_object_mask_area:
            item["accepted_for_mask"] = False
            item["reject_reason"] = "initial_mask_area_too_small"
            cluster_debug.append(item)
            continue

        # =====================================================
        # NEW: merged cluster split
        # =====================================================
        sub_masks: List[np.ndarray] = [mask_i]
        split_info: Dict[str, Any] = {
            "split_applied": False,
            "split_reason": "not_checked",
            "num_submasks": 1,
        }

        if mask_cfg.enable_split_merged_cluster:
            suspicious, suspicious_info = is_merged_cluster_candidate(
                obj_pcd=obj_i_pcd,
                mask_i=mask_i,
                cfg=mask_cfg,
            )
            item.update({f"suspicious_{k}": v for k, v in suspicious_info.items()})
            item["suspicious_for_split"] = bool(suspicious)

            if suspicious:
                sub_masks, split_info = split_cluster_by_height_watershed(
                    cluster_id=i,
                    color_bgr=color_bgr,
                    xyz=xyz,
                    mask_i=mask_i,
                    n=n,
                    d=d,
                    cfg=mask_cfg,
                )

        item.update(split_info)

        accepted_sub_count = 0

        for sub_j, sub_mask in enumerate(sub_masks, start=1):
            sub_mask = postprocess_single_object_mask(sub_mask, mask_cfg)
            sub_area = int((sub_mask > 0).sum())

            sub_item = {
                "cluster_id": i,
                "sub_id": sub_j,
                "sub_mask_area": sub_area,
            }

            if sub_area < mask_cfg.min_object_mask_area:
                sub_item["accepted_for_mask"] = False
                sub_item["reject_reason"] = "sub_mask_area_too_small"
                cluster_debug.append(sub_item)
                continue
            
            keep_support, support_info = filter_mask_by_single_frame_table_support(
                sub_mask=sub_mask,
                xyz=xyz,
                table_mask=table_mask,
                n=n,
                d=d,
                cfg=mask_cfg,
            )
            sub_item.update(support_info)

            if not keep_support:
                sub_item["accepted_for_mask"] = False
                sub_item["reject_reason"] = support_info.get("reject_reason", "support_filter_failed")
                cluster_debug.append(sub_item)
                continue

            sub_pcd = make_pcd_from_xyz_mask_and_reference_mask(
                color_bgr=color_bgr,
                xyz=xyz,
                submask=sub_mask,
                reference_mask=object_pixels_mask,
            )

            num_sub_points = int(len(sub_pcd.points))
            sub_item["sub_points"] = num_sub_points

            if num_sub_points < mask_cfg.split_min_subcluster_points:
                sub_item["accepted_for_mask"] = False
                sub_item["reject_reason"] = "subcluster_points_too_small"
                cluster_debug.append(sub_item)
                continue

            saved_objects += 1
            obj_pcd_path = os.path.join(
                mask_cfg.save_dir,
                f"{mask_cfg.object_pcd_prefix}{saved_objects}.ply"
            )
            save_pointcloud(obj_pcd_path, sub_pcd)
            object_pcd_paths.append(obj_pcd_path)

            saved_masks += 1
            mask_path = os.path.join(
                mask_cfg.save_dir,
                f"{mask_cfg.mask_prefix}{saved_masks}.png"
            )
            cv2.imwrite(mask_path, sub_mask)

            masks.append(sub_mask)
            mask_paths.append(mask_path)

            sub_item["accepted_for_mask"] = True
            sub_item["mask_path"] = mask_path
            sub_item["pcd_path"] = obj_pcd_path
            cluster_debug.append(sub_item)
            accepted_sub_count += 1

        item["accepted_sub_count"] = accepted_sub_count

    return masks, {
        "status": "ok",
        "saved_masks": saved_masks,
        "saved_objects": saved_objects,
        "table_pcd_path": table_pcd_path,
        "object_pcd_paths": object_pcd_paths,
        "mask_paths": mask_paths,
        "plane_path": save_plane_path,
        "points_table": int(len(table_pcd.points)),
        "points_objects_all": int(len(pcd_obj_all.points)),
        "cluster_debug": cluster_debug,
        "plane_flipped": plane_orient_info.get("plane_flipped", False),
        "candidate_count_original": plane_orient_info.get("candidate_count_original", 0),
        "candidate_count_flipped": plane_orient_info.get("candidate_count_flipped", 0),
        **info_seg,
    }
