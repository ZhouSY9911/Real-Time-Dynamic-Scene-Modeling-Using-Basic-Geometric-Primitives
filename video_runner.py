from __future__ import annotations

import os
import json
import numpy as np
import cv2
import pyrealsense2 as rs

from get_mask import (
    first_frame_get_and_save_masks,
    make_pointcloud_from_rgbd,
    MaskGenConfig,
)
from table_segmentation import TableSegConfig


def read_first_valid_frame_from_bag(
    bag_path: str,
    align_to_color: bool = True,
    warmup_frames: int = 10,
):
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f".bag 文件不存在: {bag_path}")

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color) if align_to_color else None
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    color_bgr = None
    depth_u16 = None
    K = None

    try:
        valid_count = 0
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                break

            if align is not None:
                frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            valid_count += 1
            if valid_count <= warmup_frames:
                continue

            depth_u16 = np.asanyarray(depth_frame.get_data()).copy()
            color_bgr = np.asanyarray(color_frame.get_data()).copy()

            intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            K = np.array(
                [
                    [intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            break
    finally:
        pipeline.stop()

    if color_bgr is None or depth_u16 is None or K is None:
        raise RuntimeError("未能从 .bag 中读取到有效的第一帧 color/depth/K")

    return color_bgr, depth_u16, depth_scale, K


def save_debug_images(
    save_dir: str,
    color_bgr: np.ndarray,
    depth_u16: np.ndarray,
):
    os.makedirs(save_dir, exist_ok=True)

    color_path = os.path.join(save_dir, "first_frame_color.png")
    cv2.imwrite(color_path, color_bgr)

    depth = depth_u16.astype(np.float32)
    valid = depth > 0
    depth_vis = np.zeros_like(depth, dtype=np.uint8)

    if np.any(valid):
        dmin = depth[valid].min()
        dmax = depth[valid].max()
        depth_norm = (depth - dmin) / (dmax - dmin + 1e-6)
        depth_vis = (depth_norm * 255).clip(0, 255).astype(np.uint8)

    depth_vis_path = os.path.join(save_dir, "first_frame_depth_vis.png")
    cv2.imwrite(depth_vis_path, depth_vis)

    return color_path, depth_vis_path


def save_overlay(
    save_dir: str,
    color_bgr: np.ndarray,
    masks: list[np.ndarray],
):
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

    for i, mask in enumerate(masks, start=1):
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
                str(i),
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


def save_run_summary(save_dir: str, info: dict):
    summary_path = os.path.join(save_dir, "run_summary.json")

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        return obj

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2, default=convert)

    return summary_path


def main():
    # =========================
    # 改这里：输入输出路径
    # =========================
    bag_path = "/mnt/data_backup/Workplace/Realtime_Dynamic_Scene/video_set/video5.bag"
    save_dir = "/mnt/data_backup/Workplace/Realtime_Dynamic_Scene/output/video5_result"
    plane_path = os.path.join(save_dir, "table_plane.npy")
    warmup_frames = 10

    # =========================
    # 桌面分割参数
    # =========================
    table_cfg = TableSegConfig(
        voxel_size=0.005,
        remove_stat_outlier=False,
        nb_neighbors=30,
        std_ratio=2.0,
        distance_threshold=0.01,
        num_iterations=3000,
        min_inlier_ratio=0.10,
        table_thickness=0.012,
    )

    # =========================
    # mask / object 提取参数
    # =========================
    mask_cfg = MaskGenConfig(
        table_band_thickness=0.012,
        table_close_ksize=21,
        table_dilate_ksize=3,
        table_erode_for_objects_ksize=21,
        table_min_area_px=5000,

        object_plane_clearance=0.015,
        object_max_distance=0.60,
        object_open_ksize=3,
        object_close_ksize=3,
        object_dilate_ksize=1,
        object_min_area_px=400,

        dbscan_eps=0.015,
        dbscan_min_points=80,
        min_cluster_points=200,

        object_mask_close_ksize=3,
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
        support_check_min_bottom_pixels=80,
        support_check_min_inside_ratio=0.60,
        edge_margin_px=12,
        edge_reject_if_bottom_inside_ratio_below=0.75,

        save_dir=save_dir,
        table_pcd_name="table.ply",
        object_pcd_prefix="object_",
        mask_prefix="obj_mask_",
    )

    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] 读取 bag: {bag_path}")
    print(f"[INFO] 输出目录: {save_dir}")

    color_bgr, depth_u16, depth_scale, K = read_first_valid_frame_from_bag(
        bag_path,
        align_to_color=True,
        warmup_frames=warmup_frames,
    )

    color_path, depth_vis_path = save_debug_images(save_dir, color_bgr, depth_u16)
    print(f"[INFO] 已保存彩色图: {color_path}")
    print(f"[INFO] 已保存深度可视化: {depth_vis_path}")

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

    overlay_path = None
    if len(masks) > 0:
        overlay_path = save_overlay(save_dir, color_bgr, masks)
        print(f"[INFO] 已保存 overlay: {overlay_path}")
    else:
        print("[WARN] 没有生成任何 mask")

    info["bag_path"] = bag_path
    info["save_dir"] = save_dir
    info["first_frame_color_path"] = color_path
    info["first_frame_depth_vis_path"] = depth_vis_path
    info["mask_overlay_path"] = overlay_path

    print("\n[RESULT]")
    for k, v in info.items():
        if k == "cluster_debug":
            continue
        print(f"{k}: {v}")

    print("\n[OUTPUT]")
    print(f"table: {info.get('table_pcd_path')}")
    for i, p in enumerate(info.get("object_pcd_paths", []), start=1):
        print(f"object_{i}: {p}")
    for i, p in enumerate(info.get("mask_paths", []), start=1):
        print(f"mask_{i}: {p}")

    summary_path = save_run_summary(save_dir, info)
    print(f"\n[INFO] 已保存运行摘要: {summary_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()