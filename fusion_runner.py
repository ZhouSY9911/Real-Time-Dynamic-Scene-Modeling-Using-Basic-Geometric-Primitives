from __future__ import annotations
import os
import sys
import json
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import cv2
import pyrealsense2 as rs

from orbslam3_runner import ORBSLAM3Runner
from get_mask import (
    first_frame_get_and_save_masks,
    make_pointcloud_from_rgbd,
    MaskGenConfig,
)
from table_segmentation import TableSegConfig

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

    bag_path: str = "/mnt/data_backup/Workplace/Realtime_Dynamic_Scene/video_set/video6.bag"
    save_root: str = "/mnt/data_backup/Workplace/Realtime_Dynamic_Scene/output/video6_keyframe_objects"

    # 是否只处理前 N 个关键帧；None 表示处理全部
    max_process_keyframes: int | None = None

    keyframe_debug_dirname: str = "keyframe_debug"

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


def save_overlay(save_dir: str, color_bgr: np.ndarray, masks: List[np.ndarray]):
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


# =========================================================
# Keyframe extraction
# =========================================================
def extract_keyframes_from_bag(
    bag_path: str,
    slam: ORBSLAM3Runner,
    cfg: FusionConfig,
) -> List[Dict[str, Any]]:
    pipeline = rs.pipeline()
    config = rs.config()

    rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
    config.enable_stream(rs.stream.depth)
    config.enable_stream(rs.stream.color)

    profile = pipeline.start(config)
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    keyframes = []
    frame_idx = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames()
            except RuntimeError:
                break

            frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color_bgr = np.asanyarray(color_frame.get_data()).copy()
            depth_u16 = np.asanyarray(depth_frame.get_data()).copy()

            intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
            K = np.array(
                [
                    [intr.fx, 0, intr.ppx],
                    [0, intr.fy, intr.ppy],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )

            timestamp = float(color_frame.get_timestamp()) * 1e-3

            if frame_idx < cfg.warmup_frames:
                frame_idx += 1
                continue

            T_wc, is_kf = slam.track(color_bgr, depth_u16, depth_scale, timestamp)
            if T_wc is None:
                frame_idx += 1
                continue

            if is_kf:
                keyframes.append(
                    {
                        "color_bgr": color_bgr,
                        "depth_u16": depth_u16,
                        "depth_scale": depth_scale,
                        "K": K,
                        "T_wc": T_wc,
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                    }
                )
                print(f"[KF] save keyframe {len(keyframes)-1:04d} <- frame {frame_idx}")

                if len(keyframes) >= cfg.max_keyframes:
                    break

            frame_idx += 1

    finally:
        pipeline.stop()

    return keyframes


# =========================================================
# Per-keyframe processing (same logic as video_runner.py)
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
        table_erode_for_objects_ksize=21,
        table_min_area_px=5000,

        object_plane_clearance=0.015,
        object_max_distance=0.60,
        object_open_ksize=3,
        object_close_ksize=7,
        object_dilate_ksize=3,
        object_min_area_px=400,

        dbscan_eps=0.025,
        dbscan_min_points=120,
        min_cluster_points=350,

        object_mask_close_ksize=7,
        object_mask_dilate_ksize=3,
        min_object_mask_area=400,

        save_dir=save_dir,
        table_pcd_name="table.ply",
        object_pcd_prefix="object_",
        mask_prefix="obj_mask_",
    )


def process_one_keyframe(
    frame: Dict[str, Any],
    kf_index: int,
    save_root: str,
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

    overlay_path = None
    if len(masks) > 0:
        overlay_path = save_overlay(kf_dir, color_bgr, masks)

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
        print(f"\n[INFO] processing keyframe {kf_index:04d} / {len(keyframes)-1:04d} "
              f"(frame_idx={int(frame['frame_idx'])})")
        info = process_one_keyframe(
            frame=frame,
            kf_index=kf_index,
            save_root=keyframe_root,
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
        cfg=cfg,
    )
    print(f"[INFO] extracted {len(keyframes)} keyframes")

    if len(keyframes) == 0:
        raise RuntimeError("No valid keyframes extracted from bag")

    process_all_keyframes(
        keyframes=keyframes,
        cfg=cfg,
    )

    print("\n[DONE]")


if __name__ == "__main__":
    main()