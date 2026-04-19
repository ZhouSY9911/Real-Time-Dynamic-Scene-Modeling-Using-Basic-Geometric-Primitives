from __future__ import annotations
import os
import sys
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
import cv2
import open3d as o3d

from orbslam3_runner import ORBSLAM3Runner
from get_mask import (
    first_frame_get_and_save_masks,
    make_pointcloud_from_rgbd,
    MaskGenConfig,
)
from table_segmentation import TableSegConfig
from extract_keyframe import extract_keyframes_from_bag
from secondary_split import (
    SecondarySplitConfig,
    refine_objects_by_pointcloud_split,
    load_pcd_points_and_colors,
    save_pcd_points_and_colors,
)

from object_id_manager import (
    FixedIDAssocConfig,
    extract_object_signature,
    initialize_global_templates_from_first_frame,
    assign_fixed_global_ids_one_frame,
    update_templates_with_frame_objects,
    build_keyframe_global_map,
)


# 让 python 找到 orbslam3.so
sys.path.insert(0, "/mnt/data_backup/Workplace/Realtime_Dynamic_Scene/src/slam3_wrapper/build")
import orbslam3


# =========================================================
# Config
# =========================================================
@dataclass
class FusionConfig:
    warmup_frames: int = 10
    max_keyframes: int = 300

    bag_path: str = "/mnt/data_backup/Workplace/Realtime_Dynamic_Scene/video_set/video7.bag"
    save_root: str = "/mnt/data_backup/Workplace/Realtime_Dynamic_Scene/output/video7.1_keyframe_objects"

    # 是否只处理前 N 个关键帧；None 表示处理全部
    max_process_keyframes: Optional[int] = None
    keyframe_debug_dirname: str = "keyframe_debug"

    # ---------- fused cloud ----------
    fused_voxel_size: float = 0.003
    fused_nb_neighbors: int = 30
    fused_std_ratio: float = 1.5
    fused_dbscan_eps: float = 0.012
    fused_dbscan_min_points: int = 20
    fused_keep_cluster_ratio: float = 0.15

    # ---------- secondary split config ----------
    secondary_split: SecondarySplitConfig = field(default_factory=SecondarySplitConfig)
    fixed_id_assoc: FixedIDAssocConfig = field(default_factory=FixedIDAssocConfig)

# =========================================================
# Basic utils
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: dict):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=convert)


def save_debug_images(save_dir: str, color_bgr: np.ndarray, depth_u16: np.ndarray):
    ensure_dir(save_dir)

    color_path = os.path.join(save_dir, "color.png")
    cv2.imwrite(color_path, color_bgr)

    depth = depth_u16.astype(np.float32)
    valid = depth > 0
    depth_vis = np.zeros_like(depth, dtype=np.uint8)
    if np.any(valid):
        dmin = depth[valid].min()
        dmax = depth[valid].max()
        depth_norm = (depth - dmin) / (dmax - dmin + 1e-6)
        depth_vis = (depth_norm * 255).clip(0, 255).astype(np.uint8)

    depth_vis_path = os.path.join(save_dir, "depth_vis.png")
    cv2.imwrite(depth_vis_path, depth_vis)

    return color_path, depth_vis_path


def save_overlay(save_dir: str, color_bgr: np.ndarray, masks: List[np.ndarray], ids: List[int] | None = None):
    overlay = color_bgr.copy()

    color_table = [
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 0, 255),
        (0, 128, 255),
        (255, 255, 0),
        (128, 0, 255),
        (255, 128, 0),
    ]

    if ids is None:
        ids = list(range(1, len(masks) + 1))

    for i, (mask, obj_id) in enumerate(zip(masks, ids), start=1):
        idx = mask > 0
        if not np.any(idx):
            continue

        bgr = color_table[(i - 1) % len(color_table)]
        color_layer = np.zeros_like(overlay)
        color_layer[:, :, 0] = bgr[0]
        color_layer[:, :, 1] = bgr[1]
        color_layer[:, :, 2] = bgr[2]

        alpha = 0.35
        overlay[idx] = (
            overlay[idx].astype(np.float32) * (1.0 - alpha)
            + color_layer[idx].astype(np.float32) * alpha
        ).astype(np.uint8)

        ys, xs = np.where(idx)
        if len(xs) > 0 and len(ys) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(
                overlay,
                str(obj_id),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    overlay_path = os.path.join(save_dir, "mask_overlay.png")
    cv2.imwrite(overlay_path, overlay)
    return overlay_path

def save_overlay_from_mask_paths(
    save_dir: str,
    color_bgr: np.ndarray,
    mask_paths: List[str],
    ids: List[int] | None = None,
    out_name: str = "mask_overlay.png",
):
    masks = []
    valid_ids = []

    if ids is None:
        ids = list(range(1, len(mask_paths) + 1))

    for p, obj_id in zip(mask_paths, ids):
        if not os.path.exists(p):
            continue
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            masks.append(m)
            valid_ids.append(obj_id)

    if len(masks) == 0:
        return None

    overlay = color_bgr.copy()

    color_table = [
        (0, 255, 0),
        (0, 255, 255),
        (255, 0, 0),
        (255, 0, 255),
        (0, 128, 255),
        (255, 255, 0),
        (128, 0, 255),
        (255, 128, 0),
    ]

    for i, (mask, obj_id) in enumerate(zip(masks, valid_ids), start=1):
        idx = mask > 0
        if not np.any(idx):
            continue

        bgr = color_table[(i - 1) % len(color_table)]
        color_layer = np.zeros_like(overlay)
        color_layer[:, :, 0] = bgr[0]
        color_layer[:, :, 1] = bgr[1]
        color_layer[:, :, 2] = bgr[2]

        alpha = 0.35
        overlay[idx] = (
            overlay[idx].astype(np.float32) * (1.0 - alpha)
            + color_layer[idx].astype(np.float32) * alpha
        ).astype(np.uint8)

        ys, xs = np.where(idx)
        if len(xs) > 0 and len(ys) > 0:
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(
                overlay,
                str(obj_id),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    overlay_path = os.path.join(save_dir, out_name)
    cv2.imwrite(overlay_path, overlay)
    return overlay_path


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts.copy()
    pts_h = np.concatenate(
        [pts.astype(np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)],
        axis=1
    )
    out = (T @ pts_h.T).T
    return out[:, :3]


def denoise_fused_points_and_colors(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float = 0.003,
    nb_neighbors: int = 30,
    std_ratio: float = 1.5,
    dbscan_eps: float = 0.012,
    dbscan_min_points: int = 20,
    keep_cluster_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return points, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and colors.shape[0] == points.shape[0]:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if voxel_size is not None and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    try:
        if len(pcd.points) >= nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
    except Exception:
        pass

    try:
        labels = np.array(
            pcd.cluster_dbscan(
                eps=dbscan_eps,
                min_points=dbscan_min_points,
                print_progress=False
            )
        )

        if labels.size > 0 and np.any(labels >= 0):
            valid = labels[labels >= 0]
            counts = np.bincount(valid)
            max_count = counts.max()

            keep_idx = []
            for lab, cnt in enumerate(counts):
                if cnt >= max_count * keep_cluster_ratio:
                    keep_idx.extend(np.where(labels == lab)[0].tolist())

            if len(keep_idx) > 0:
                pcd = pcd.select_by_index(keep_idx)
    except Exception:
        pass

    pts = np.asarray(pcd.points).astype(np.float32)
    cols = np.asarray(pcd.colors).astype(np.float32)

    if cols.shape[0] != pts.shape[0]:
        cols = np.zeros((pts.shape[0], 3), dtype=np.float32)

    return pts, cols


# =========================================================
# Association and fusion
# =========================================================
def associate_and_fuse_objects(
    keyframes: List[Dict[str, Any]],
    infos: List[Dict[str, Any]],
    cfg: FusionConfig,
):
    fused_dir = os.path.join(cfg.save_root, "fused_objects")
    ensure_dir(fused_dir)

    if len(keyframes) == 0:
        return

    # 用第一个关键帧相机系作为统一参考系
    T_wc0 = np.asarray(keyframes[0]["T_wc"], dtype=np.float32)
    T_c0w = np.linalg.inv(T_wc0)

    templates = []
    per_keyframe_global_map = []

    for frame_idx_in_list, (frame, info) in enumerate(zip(keyframes, infos)):
        T_wc = np.asarray(frame["T_wc"], dtype=np.float32)
        object_pcd_paths = info.get("object_pcd_paths", [])
        mask_paths = info.get("mask_paths", [])

        frame_objects = []

        num_objs = min(len(object_pcd_paths), len(mask_paths))
        for i in range(num_objs):
            obj_path = object_pcd_paths[i]
            mask_path = mask_paths[i]

            if not os.path.exists(obj_path):
                continue

            pts_c, cols_c = load_pcd_points_and_colors(obj_path)
            if pts_c.shape[0] == 0:
                continue

            # 当前相机系 -> 世界系 -> 第一关键帧参考系
            pts_w = transform_points(T_wc, pts_c)
            pts_ref = transform_points(T_c0w, pts_w)

            sig = extract_object_signature(pts_ref, cols_c)

            frame_objects.append({
                "local_id": i + 1,
                "pcd_path": obj_path,
                "mask_path": mask_path,
                "points_ref": pts_ref,
                "colors": cols_c,
                "signature": sig,
                "keyframe_index": int(info["keyframe_index"]),
                "frame_idx": int(info["frame_idx"]),
            })

        # 第一帧：直接初始化 global_id 模板
        if frame_idx_in_list == 0:
            templates = initialize_global_templates_from_first_frame(frame_objects)
            assigned_global_ids = [int(t["global_id"]) for t in templates[:len(frame_objects)]]
        else:
            # 后续帧：只匹配第一帧模板，不允许新增 global_id
            assigned_global_ids = assign_fixed_global_ids_one_frame(
                frame_objects=frame_objects,
                templates=templates,
                cfg=cfg.fixed_id_assoc,
            )

            update_templates_with_frame_objects(
                frame_objects=frame_objects,
                assigned_global_ids=assigned_global_ids,
                templates=templates,
            )

        # 保存每帧 local_id -> global_id 映射
        one_kf_map = build_keyframe_global_map(
            info=info,
            frame_objects=frame_objects,
            assigned_global_ids=assigned_global_ids,
        )
        per_keyframe_global_map.append(one_kf_map)

        # 生成 global_id overlay，方便检查每帧 id 是否稳定
        kf_dir = info.get("kf_dir", "")
        color_path = info.get("color_path", "")
        if kf_dir and os.path.exists(color_path):
            color_bgr = cv2.imread(color_path, cv2.IMREAD_COLOR)
            if color_bgr is not None:
                frame_mask_paths = [obj["mask_path"] for obj in frame_objects]
                save_overlay_from_mask_paths(
                    save_dir=kf_dir,
                    color_bgr=color_bgr,
                    mask_paths=frame_mask_paths,
                    ids=assigned_global_ids,
                    out_name="mask_overlay_global_id.png",
                )

    # 保存映射关系
    save_json(os.path.join(cfg.save_root, "global_id_map.json"), {
        "status": "ok",
        "num_global_objects": len(templates),
        "keyframes": per_keyframe_global_map,
    })

    # 最终按固定 global_id 融合
    summary_tracks = []
    for tpl in templates:
        global_id = int(tpl["global_id"])

        all_pts = np.concatenate(tpl["points_list"], axis=0)
        all_cols = np.concatenate(tpl["colors_list"], axis=0)

        all_pts, all_cols = denoise_fused_points_and_colors(
            all_pts,
            all_cols,
            voxel_size=cfg.fused_voxel_size,
            nb_neighbors=cfg.fused_nb_neighbors,
            std_ratio=cfg.fused_std_ratio,
            dbscan_eps=cfg.fused_dbscan_eps,
            dbscan_min_points=cfg.fused_dbscan_min_points,
            keep_cluster_ratio=cfg.fused_keep_cluster_ratio,
        )

        save_path = os.path.join(fused_dir, f"global_object_{global_id:02d}_fused.ply")
        save_pcd_points_and_colors(save_path, all_pts, all_cols)

        summary_tracks.append({
            "global_id": global_id,
            "num_points": int(all_pts.shape[0]),
            "num_members": len(tpl["members"]),
            "members": tpl["members"],
            "fused_pcd_path": save_path,
            "template_center": tpl["template_signature"]["center"],
            "template_extent": tpl["template_signature"]["extent"],
            "template_mean_color": tpl["template_signature"]["mean_color"],
        })

        print(f"[FUSE] global_object_{global_id:02d}: {len(tpl['members'])} members -> {save_path}")

    save_json(os.path.join(cfg.save_root, "fusion_summary.json"), {
        "status": "ok",
        "num_keyframes": len(keyframes),
        "num_global_objects": len(summary_tracks),
        "tracks": summary_tracks,
    })


# =========================================================
# Per-keyframe processing
# =========================================================
def build_table_cfg() -> TableSegConfig:
    return TableSegConfig(
        voxel_size=0.005,
        remove_stat_outlier=False,
        nb_neighbors=30,
        std_ratio=2.0,
        distance_threshold=0.01,
        num_iterations=3000,
        min_inlier_ratio=0.10,
        table_thickness=0.012,
    )


def build_mask_cfg(save_dir: str) -> MaskGenConfig:
    return MaskGenConfig(
        table_band_thickness=0.012,
        table_close_ksize=21,
        table_dilate_ksize=3,
        table_erode_for_objects_ksize=9,
        table_min_area_px=5000,

        object_plane_clearance=0.015,
        object_max_distance=0.60,
        object_open_ksize=3,
        object_close_ksize=3,
        object_dilate_ksize=1,
        object_min_area_px=400,

        dbscan_eps=0.015,
        dbscan_min_points=80,
        min_cluster_points=150,

        object_mask_close_ksize=5,
        object_mask_dilate_ksize=1,
        min_object_mask_area=400,

        enable_split_merged_cluster=True,
        split_check_min_points=1200,
        split_check_min_mask_area=2500,
        split_check_bbox_max_side_px=120,
        split_gaussian_ksize=5,
        split_dist_thresh_ratio=0.38,
        split_height_percentile_clip_low=5.0,
        split_height_percentile_clip_high=99.0,
        split_min_peak_components=2,
        split_min_submask_area=300,
        split_min_subcluster_points=150,
        save_split_debug=True,

        support_check_bottom_band=0.02,
        support_check_min_bottom_pixels=40,
        support_check_min_inside_ratio=0.45,
        edge_margin_px=12,
        edge_reject_if_bottom_inside_ratio_below=0.60,


        save_dir=save_dir,
        table_pcd_name="table.ply",
        object_pcd_prefix="object_",
        mask_prefix="obj_mask_",
    )


def process_one_keyframe(
    frame: Dict[str, Any],
    kf_index: int,
    save_root: str,
    cfg: FusionConfig,
) -> Dict[str, Any]:
    kf_dir = os.path.join(save_root, f"kf_{kf_index:04d}")
    ensure_dir(kf_dir)

    color_bgr = frame["color_bgr"]
    depth_u16 = frame["depth_u16"]
    depth_scale = frame["depth_scale"]
    K = frame["K"]

    color_path, depth_vis_path = save_debug_images(kf_dir, color_bgr, depth_u16)

    table_cfg = build_table_cfg()
    mask_cfg = build_mask_cfg(kf_dir)
    plane_path = os.path.join(kf_dir, "table_plane.npy")

    masks, info = first_frame_get_and_save_masks(
        color_bgr=color_bgr,
        depth_u16=depth_u16,
        depth_scale=depth_scale,
        K=K,
        make_pointcloud_fn=make_pointcloud_from_rgbd,
        table_cfg=table_cfg,
        mask_cfg=mask_cfg,
        save_plane_path=plane_path,
    )

    depth_map = depth_u16.astype(np.float32) * float(depth_scale)

    # 基于点云 + 深度图做二次分割
    info = refine_objects_by_pointcloud_split(
        info=info,
        K=K,
        image_shape=color_bgr.shape[:2],
        save_dir=kf_dir,
        cfg=cfg.secondary_split,
        depth_map=depth_map,
    )


    # =====================================================
    # 只保留最终 refined 结果，并统一重命名
    # =====================================================
    refined_pcd_paths = info.get("object_pcd_paths", [])
    refined_mask_paths = info.get("mask_paths", [])

    final_pcd_paths = []
    final_mask_paths = []

    num_objs = min(len(refined_pcd_paths), len(refined_mask_paths))
    for i in range(num_objs):
        src_pcd = refined_pcd_paths[i]
        src_mask = refined_mask_paths[i]

        if not os.path.exists(src_pcd) or not os.path.exists(src_mask):
            continue

        pts, cols = load_pcd_points_and_colors(src_pcd)
        mask = cv2.imread(src_mask, cv2.IMREAD_GRAYSCALE)

        if pts.shape[0] == 0 or mask is None:
            continue

        final_pcd_path = os.path.join(kf_dir, f"object_{len(final_pcd_paths)+1}.ply")
        final_mask_path = os.path.join(kf_dir, f"obj_mask_{len(final_mask_paths)+1}.png")

        save_pcd_points_and_colors(final_pcd_path, pts, cols)
        cv2.imwrite(final_mask_path, mask)

        final_pcd_paths.append(final_pcd_path)
        final_mask_paths.append(final_mask_path)

    # 删除中间 refined / split 文件
    for p in refined_pcd_paths:
        if os.path.exists(p) and p not in final_pcd_paths:
            try:
                os.remove(p)
            except Exception:
                pass

    for p in refined_mask_paths:
        if os.path.exists(p) and p not in final_mask_paths:
            try:
                os.remove(p)
            except Exception:
                pass

    info["object_pcd_paths"] = final_pcd_paths
    info["mask_paths"] = final_mask_paths
    info["saved_objects"] = len(final_pcd_paths)
    info["saved_masks"] = len(final_mask_paths)

    overlay_path = None
    if len(final_mask_paths) > 0:
        overlay_path = save_overlay_from_mask_paths(kf_dir, color_bgr, final_mask_paths)

    info["keyframe_index"] = kf_index
    info["frame_idx"] = int(frame["frame_idx"])
    info["timestamp"] = float(frame["timestamp"])
    info["color_path"] = color_path
    info["depth_vis_path"] = depth_vis_path
    info["overlay_path"] = overlay_path
    info["kf_dir"] = kf_dir

    save_json(os.path.join(kf_dir, "run_summary.json"), info)
    return info


def process_all_keyframes(
    keyframes: List[Dict[str, Any]],
    cfg: FusionConfig,
):
    keyframe_root = os.path.join(cfg.save_root, cfg.keyframe_debug_dirname)
    ensure_dir(keyframe_root)

    if cfg.max_process_keyframes is not None:
        keyframes = keyframes[:cfg.max_process_keyframes]

    all_infos = []
    for kf_index, frame in enumerate(keyframes):
        print(
            f"\n[INFO] processing keyframe {kf_index:04d} / {len(keyframes)-1:04d} "
            f"(frame_idx={int(frame['frame_idx'])})"
        )
        info = process_one_keyframe(
            frame=frame,
            kf_index=kf_index,
            save_root=keyframe_root,
            cfg=cfg,
        )
        all_infos.append(info)

        print(f"status: {info.get('status')}")
        print(f"saved_objects: {info.get('saved_objects')}")
        print(f"saved_masks: {info.get('saved_masks')}")

    save_json(os.path.join(cfg.save_root, "all_keyframes_summary.json"), {
        "status": "ok",
        "num_keyframes_processed": len(all_infos),
        "keyframes": all_infos,
    })

    return all_infos


# =========================================================
# Main
# =========================================================
def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    vocab_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "..", "third_party", "ORB_SLAM3", "Vocabulary", "ORBvoc.txt")
    )
    settings_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "configs", "RealSense_D435.yaml")
    )

    cfg = FusionConfig()
    ensure_dir(cfg.save_root)

    slam = ORBSLAM3Runner(
        orbslam3,
        vocab_path=vocab_path,
        settings_path=settings_path,
        use_viewer=False,
    )

    print(f"[INFO] reading bag: {cfg.bag_path}")
    print("[INFO] extracting keyframes...")
    keyframes = extract_keyframes_from_bag(
        bag_path=cfg.bag_path,
        slam=slam,
        warmup_frames=cfg.warmup_frames,
        max_keyframes=cfg.max_keyframes,
    )
    print(f"[INFO] extracted {len(keyframes)} keyframes")

    if len(keyframes) == 0:
        raise RuntimeError("No valid keyframes extracted from bag")

    infos = process_all_keyframes(
        keyframes=keyframes,
        cfg=cfg,
    )

    print("\n[INFO] associating and fusing objects...")
    associate_and_fuse_objects(
        keyframes=keyframes if cfg.max_process_keyframes is None else keyframes[:cfg.max_process_keyframes],
        infos=infos,
        cfg=cfg,
    )

    print("\n[DONE]")


if __name__ == "__main__":
    main()