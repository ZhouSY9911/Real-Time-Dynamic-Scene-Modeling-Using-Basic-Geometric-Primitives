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
    object_close_ksize: int = 7
    object_dilate_ksize: int = 3
    object_min_area_px: int = 400

    # ---------- clustering ----------
    dbscan_eps: float = 0.025
    dbscan_min_points: int = 120
    min_cluster_points: int = 350

    # ---------- single object mask ----------
    object_mask_close_ksize: int = 7
    object_mask_dilate_ksize: int = 3
    min_object_mask_area: int = 400

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
        (np.abs(signed_dist) >= cfg.object_plane_clearance) &
        (np.abs(signed_dist) <= cfg.object_max_distance)
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


def cluster_extent_ok(obj_pcd: o3d.geometry.PointCloud, n: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    pts = np.asarray(obj_pcd.points)
    if pts.shape[0] == 0:
        return False, {"reason": -1.0}

    bbox = obj_pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent(), dtype=np.float64)
    max_extent = float(extent.max())

    n = _normalize(n)
    hh = pts @ n
    height_extent = float(hh.max() - hh.min())
    center_height = float(np.median(np.abs(hh)))

    # 固定过滤规则，避免 video_runner 传参不匹配
    ok = True
    if max_extent < 0.01:
        ok = False
    if max_extent > 0.50:
        ok = False
    if height_extent > 0.50:
        ok = False
    if center_height < 0.02:
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

    save_plane_model(save_plane_path, n, d)

    xyz = build_xyz_map_from_depth(depth_u16, depth_scale, K)

    # 3) 桌面mask
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

    # 7) 聚类
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

        ok_cluster, extent_info = cluster_extent_ok(obj_i_pcd, n)
        item = {
            "cluster_id": i,
            "num_points": int(len(obj_i_pcd.points)),
            **extent_info,
            "accepted": bool(ok_cluster),
        }
        if not ok_cluster:
            cluster_debug.append(item)
            continue

        saved_objects += 1
        obj_pcd_path = os.path.join(mask_cfg.save_dir, f"{mask_cfg.object_pcd_prefix}{saved_objects}.ply")
        save_pointcloud(obj_pcd_path, obj_i_pcd)
        object_pcd_paths.append(obj_pcd_path)

        pts_i = pts_all[idx]
        mask_i = project_points_to_mask(pts_i, K, H, W)
        mask_i = postprocess_single_object_mask(mask_i, mask_cfg)

        area = int((mask_i > 0).sum())
        item["mask_area"] = area

        if area < mask_cfg.min_object_mask_area:
            item["accepted_for_mask"] = False
            cluster_debug.append(item)
            continue

        saved_masks += 1
        mask_path = os.path.join(mask_cfg.save_dir, f"{mask_cfg.mask_prefix}{saved_masks}.png")
        cv2.imwrite(mask_path, mask_i)

        masks.append(mask_i)
        mask_paths.append(mask_path)

        item["accepted_for_mask"] = True
        item["mask_path"] = mask_path
        cluster_debug.append(item)

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
        **info_seg,
    }