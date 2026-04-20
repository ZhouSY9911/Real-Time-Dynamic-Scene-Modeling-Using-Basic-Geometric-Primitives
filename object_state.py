from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import os
import json

import numpy as np
import open3d as o3d
import cv2

from secondary_split import load_pcd_points_and_colors, save_pcd_points_and_colors
from object_id_manager import (
    extract_object_signature,
    initialize_global_templates_from_first_frame,
    assign_fixed_global_ids_one_frame,
    build_keyframe_global_map,
    FixedIDAssocConfig,
)
from geometry_init import init_geometry_from_first_frame, build_init_geometry_mesh
from geometry_infer_and_update import (
    infer_and_update_geometry,
    build_updated_geometry_mesh,
)


# =========================================================
# Basic utils
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data: Dict[str, Any]):
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


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return pts.copy()

    pts_h = np.concatenate(
        [pts.astype(np.float32), np.ones((pts.shape[0], 1), dtype=np.float32)],
        axis=1
    )
    out = (T @ pts_h.T).T
    return out[:, :3]


def compute_pointcloud_center(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    return np.median(points, axis=0).astype(np.float32)


def compute_pointcloud_extent(points: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    pmin = np.percentile(points, 5, axis=0)
    pmax = np.percentile(points, 95, axis=0)
    return (pmax - pmin).astype(np.float32)


def mean_color_safe(colors: np.ndarray) -> np.ndarray:
    if colors is None or colors.shape[0] == 0:
        return np.zeros(3, dtype=np.float32)
    return np.mean(colors, axis=0).astype(np.float32)


def confidence_rank(conf: str) -> int:
    table = {"low": 0, "medium": 1, "high": 2}
    return table.get(conf, 0)


# =========================================================
# Denoise
# =========================================================
def denoise_points_and_colors(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float = 0.003,
    nb_neighbors: int = 20,
    std_ratio: float = 1.5,
    keep_largest_cluster: bool = False,
    dbscan_eps: float = 0.012,
    dbscan_min_points: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
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
                std_ratio=std_ratio,
            )
    except Exception:
        pass

    if keep_largest_cluster and len(pcd.points) > 0:
        try:
            labels = np.array(
                pcd.cluster_dbscan(
                    eps=dbscan_eps,
                    min_points=dbscan_min_points,
                    print_progress=False,
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


# =========================================================
# Config
# =========================================================
@dataclass
class ObjectStateConfig:
    # 每个关键帧单个物体先降噪一次
    observation_voxel_size: float = 0.003
    observation_nb_neighbors: int = 20
    observation_std_ratio: float = 1.5

    # 最终完整点云统一降噪
    final_voxel_size: float = 0.003
    final_nb_neighbors: int = 30
    final_std_ratio: float = 1.5
    final_keep_largest_cluster: bool = False
    final_dbscan_eps: float = 0.012
    final_dbscan_min_points: int = 20

    # geometry inference 专用降噪
    geometry_voxel_size: float = 0.003
    geometry_nb_neighbors: int = 30
    geometry_std_ratio: float = 1.5
    geometry_keep_largest_cluster: bool = False
    geometry_dbscan_eps: float = 0.012
    geometry_dbscan_min_points: int = 20

    # 单帧最低点数
    min_points_per_object: int = 50

    # 保存
    save_fused_every_keyframe: bool = True
    incremental_dirname: str = "incremental_fused_objects"
    final_dirname: str = "final_fused_objects"

    # geometry
    enable_geometry_update: bool = True
    geometry_confirm_frames: int = 3
    geometry_save_mesh_every_keyframe: bool = True
    geometry_min_confidence_to_switch: str = "medium"


# =========================================================
# Data structures
# =========================================================
@dataclass
class ObjectTrack:
    global_id: int

    # 从第一帧开始累计的点云（每帧单独降噪后追加）
    accum_points_ref: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    accum_colors_ref: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))

    obs_count: int = 0
    last_seen_keyframe: int = -1

    center_ref: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    extent_ref: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    mean_color: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    members: List[Dict[str, Any]] = field(default_factory=list)

    # geometry state
    current_geom: Any = None
    pending_geom: Any = None
    pending_count: int = 0
    geom_history: List[Dict[str, Any]] = field(default_factory=list)


# =========================================================
# Track manager
# =========================================================
class ObjectTrackManager:
    def __init__(
        self,
        save_root: str,
        fixed_id_assoc_cfg: FixedIDAssocConfig,
        state_cfg: Optional[ObjectStateConfig] = None,
    ):
        self.save_root = save_root
        self.fixed_id_assoc_cfg = fixed_id_assoc_cfg
        self.state_cfg = state_cfg or ObjectStateConfig()

        self.tracks: Dict[int, ObjectTrack] = {}
        self.templates: List[Dict[str, Any]] = []
        self.per_keyframe_global_map: List[Dict[str, Any]] = []

        self.initialized = False
        self.T_wc0: Optional[np.ndarray] = None
        self.T_c0w: Optional[np.ndarray] = None

        # 只允许第一帧已有的固定ID参与后续融合
        self.allowed_global_ids: List[int] = []

        self.incremental_dir = os.path.join(self.save_root, self.state_cfg.incremental_dirname)
        self.final_dir = os.path.join(self.save_root, self.state_cfg.final_dirname)

        ensure_dir(self.incremental_dir)
        ensure_dir(self.final_dir)

    # -----------------------------------------------------
    # Public
    # -----------------------------------------------------
    def update_with_keyframe(
        self,
        frame: Dict[str, Any],
        info: Dict[str, Any],
        kf_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.T_wc0 is None:
            self.T_wc0 = np.asarray(frame["T_wc"], dtype=np.float32)
            self.T_c0w = np.linalg.inv(self.T_wc0)

        frame_objects = self._build_frame_objects(frame, info)

        # 第一帧：初始化固定模板和固定ID
        if not self.initialized:
            self.templates = initialize_global_templates_from_first_frame(frame_objects)
            assigned_global_ids = [int(t["global_id"]) for t in self.templates[:len(frame_objects)]]
            self.allowed_global_ids = assigned_global_ids.copy()
            self.initialized = True
        else:
            assigned_global_ids = assign_fixed_global_ids_one_frame(
                frame_objects=frame_objects,
                templates=self.templates,
                cfg=self.fixed_id_assoc_cfg,
            )

        kf_map = build_keyframe_global_map(
            info=info,
            frame_objects=frame_objects,
            assigned_global_ids=assigned_global_ids,
        )
        self.per_keyframe_global_map.append(kf_map)

        update_debug = []
        kept_assigned_global_ids: List[int] = []

        for obj, gid in zip(frame_objects, assigned_global_ids):
            gid = int(gid)

            if gid < 0:
                update_debug.append({
                    "local_id": int(obj["local_id"]),
                    "global_id": gid,
                    "status": "ignored",
                    "reason": "global_id_is_minus_one",
                })
                continue

            if gid not in self.allowed_global_ids:
                update_debug.append({
                    "local_id": int(obj["local_id"]),
                    "global_id": gid,
                    "status": "ignored",
                    "reason": "not_in_allowed_global_ids",
                })
                continue

            dbg = self._update_one_object_track(
                frame_obj=obj,
                global_id=gid,
                kf_dir=kf_dir,
                info=info,
            )
            update_debug.append(dbg)

            if dbg.get("status") in ["initialized", "updated"]:
                kept_assigned_global_ids.append(gid)

        self._save_global_map()

        if kf_dir is None:
            kf_dir = info.get("kf_dir", None)

        if kf_dir:
            self._save_tracks_to_keyframe_dir(kf_dir)
            self._save_geometry_meshes_to_keyframe_dir(kf_dir)

        return {
            "status": "ok",
            "num_tracks": len(self.tracks),
            "num_frame_objects": len(frame_objects),
            "assigned_global_ids": kept_assigned_global_ids,
            "allowed_global_ids": self.allowed_global_ids,
            "update_debug": update_debug,
        }

    def save_summary(self):
        tracks_summary = []
        for gid in sorted(self.tracks.keys()):
            tr = self.tracks[gid]
            latest_path = os.path.join(self.incremental_dir, f"global_object_{gid:02d}_latest.ply")

            tracks_summary.append({
                "global_id": gid,
                "obs_count": tr.obs_count,
                "num_points_accum": int(tr.accum_points_ref.shape[0]),
                "center_ref": tr.center_ref,
                "extent_ref": tr.extent_ref,
                "mean_color": tr.mean_color,
                "accum_pcd_path": latest_path,
                "num_members": len(tr.members),
                "members": tr.members,
                "current_geom_shape": tr.current_geom.shape if tr.current_geom is not None else "none",
                "current_geom_confidence": tr.current_geom.confidence if tr.current_geom is not None else "none",
                "current_geom_params": tr.current_geom.params if tr.current_geom is not None else {},
                "pending_geom_shape": tr.pending_geom.shape if tr.pending_geom is not None else "none",
                "pending_count": int(tr.pending_count),
                "geom_history_len": len(tr.geom_history),
                "geom_history": tr.geom_history,
            })

        save_json(os.path.join(self.save_root, "object_state_summary.json"), {
            "status": "ok",
            "allowed_global_ids": self.allowed_global_ids,
            "num_tracks": len(tracks_summary),
            "tracks": tracks_summary,
        })

    def save_final_fused_objects(self):
        final_summary = []
        for gid in sorted(self.tracks.keys()):
            tr = self.tracks[gid]
            if tr.accum_points_ref.shape[0] == 0:
                continue

            final_points, final_colors = denoise_points_and_colors(
                tr.accum_points_ref,
                tr.accum_colors_ref,
                voxel_size=self.state_cfg.final_voxel_size,
                nb_neighbors=self.state_cfg.final_nb_neighbors,
                std_ratio=self.state_cfg.final_std_ratio,
                keep_largest_cluster=self.state_cfg.final_keep_largest_cluster,
                dbscan_eps=self.state_cfg.final_dbscan_eps,
                dbscan_min_points=self.state_cfg.final_dbscan_min_points,
            )

            save_path = os.path.join(self.final_dir, f"global_object_{gid:02d}_final.ply")
            save_pcd_points_and_colors(save_path, final_points, final_colors)

            mesh_path = None
            if tr.current_geom is not None:
                try:
                    mesh = self._build_mesh_for_geom(tr.current_geom)
                    if mesh is not None:
                        mesh_path = os.path.join(self.final_dir, f"global_object_{gid:02d}_final_geom.ply")
                        mesh.export(mesh_path)
                except Exception:
                    mesh_path = None

            final_summary.append({
                "global_id": gid,
                "num_points": int(final_points.shape[0]),
                "path": save_path,
                "geom_shape": tr.current_geom.shape if tr.current_geom is not None else "none",
                "geom_confidence": tr.current_geom.confidence if tr.current_geom is not None else "none",
                "geom_params": tr.current_geom.params if tr.current_geom is not None else {},
                "geom_mesh_path": mesh_path,
            })

        save_json(os.path.join(self.save_root, "final_fused_summary.json"), {
            "status": "ok",
            "allowed_global_ids": self.allowed_global_ids,
            "num_tracks": len(final_summary),
            "tracks": final_summary,
        })

    # -----------------------------------------------------
    # Internal
    # -----------------------------------------------------
    def _build_frame_objects(
        self,
        frame: Dict[str, Any],
        info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        T_wc = np.asarray(frame["T_wc"], dtype=np.float32)
        assert self.T_c0w is not None

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

            pts_w = transform_points(T_wc, pts_c)
            pts_ref = transform_points(self.T_c0w, pts_w)

            sig = extract_object_signature(pts_ref, cols_c)

            mask_img = None
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            frame_objects.append({
                "local_id": i + 1,
                "pcd_path": obj_path,
                "mask_path": mask_path,
                "mask": mask_img,
                "points_cam": pts_c,
                "points_ref": pts_ref,
                "colors": cols_c,
                "signature": sig,
                "keyframe_index": int(info["keyframe_index"]),
                "frame_idx": int(info["frame_idx"]),
            })

        return frame_objects

    def _prepare_points_for_geometry_inference(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if points.shape[0] == 0:
            return points, colors

        pts, cols = denoise_points_and_colors(
            points,
            colors,
            voxel_size=self.state_cfg.geometry_voxel_size,
            nb_neighbors=self.state_cfg.geometry_nb_neighbors,
            std_ratio=self.state_cfg.geometry_std_ratio,
            keep_largest_cluster=self.state_cfg.geometry_keep_largest_cluster,
            dbscan_eps=self.state_cfg.geometry_dbscan_eps,
            dbscan_min_points=self.state_cfg.geometry_dbscan_min_points,
        )
        return pts, cols

    def _build_mesh_for_geom(self, geom):
        if geom is None:
            return None

        try:
            if geom.shape in ["sphere", "cylinder"]:
                return build_updated_geometry_mesh(geom)

            if geom.shape == "box":
                if all(k in geom.params for k in ["sx", "sy", "sz"]):
                    return build_updated_geometry_mesh(geom)
                return build_init_geometry_mesh(geom)

            return build_updated_geometry_mesh(geom)
        except Exception:
            try:
                return build_init_geometry_mesh(geom)
            except Exception:
                return None

    def _init_track_from_object(
        self,
        frame_obj: Dict[str, Any],
        global_id: int,
    ) -> ObjectTrack:
        pts_cam_raw = frame_obj["points_cam"]
        pts_ref_raw = frame_obj["points_ref"]
        cols_raw = frame_obj["colors"]
        mask = frame_obj.get("mask", None)

        # ref 系用于累计
        pts_ref, cols_ref = denoise_points_and_colors(
            pts_ref_raw,
            cols_raw,
            voxel_size=self.state_cfg.observation_voxel_size,
            nb_neighbors=self.state_cfg.observation_nb_neighbors,
            std_ratio=self.state_cfg.observation_std_ratio,
            keep_largest_cluster=False,
        )

        # cam 系用于初始几何
        pts_cam_geom, cols_cam_geom = denoise_points_and_colors(
            pts_cam_raw,
            cols_raw,
            voxel_size=self.state_cfg.observation_voxel_size,
            nb_neighbors=self.state_cfg.observation_nb_neighbors,
            std_ratio=self.state_cfg.observation_std_ratio,
            keep_largest_cluster=False,
        )

        current_geom = None
        if (
            self.state_cfg.enable_geometry_update and
            mask is not None and
            pts_cam_geom.shape[0] >= self.state_cfg.min_points_per_object
        ):
            try:
                current_geom = init_geometry_from_first_frame(
                    pts_cam=pts_cam_geom,
                    mask=mask,
                    min_points=self.state_cfg.min_points_per_object,
                )
            except Exception:
                current_geom = None

        track = ObjectTrack(
            global_id=global_id,
            accum_points_ref=pts_ref.astype(np.float32),
            accum_colors_ref=cols_ref.astype(np.float32),
            obs_count=1,
            last_seen_keyframe=int(frame_obj["keyframe_index"]),
            center_ref=compute_pointcloud_center(pts_ref),
            extent_ref=compute_pointcloud_extent(pts_ref),
            mean_color=mean_color_safe(cols_ref),
            members=[{
                "keyframe_index": int(frame_obj["keyframe_index"]),
                "frame_idx": int(frame_obj["frame_idx"]),
                "local_id": int(frame_obj["local_id"]),
                "num_points_used": int(pts_ref.shape[0]),
            }],
            current_geom=current_geom,
            pending_geom=None,
            pending_count=0,
            geom_history=[{
                "obs_count": 1,
                "event": "init",
                "current_shape": current_geom.shape if current_geom is not None else "none",
                "current_confidence": current_geom.confidence if current_geom is not None else "none",
                "num_init_points_ref": int(pts_ref.shape[0]),
                "num_init_points_cam_geom": int(pts_cam_geom.shape[0]),
            }],
        )
        return track

    def _update_geometry_for_track(
        self,
        track: ObjectTrack,
        info: Optional[Dict[str, Any]] = None,
        kf_dir: Optional[str] = None,
    ):
        if not self.state_cfg.enable_geometry_update:
            return

        if track.current_geom is None:
            return

        if track.accum_points_ref.shape[0] < max(200, self.state_cfg.min_points_per_object):
            return

        geom_points, geom_colors = self._prepare_points_for_geometry_inference(
            track.accum_points_ref,
            track.accum_colors_ref,
        )

        min_geom_points = max(180, self.state_cfg.min_points_per_object)
        if geom_points.shape[0] < min_geom_points:
            return

        try:
            candidate_geom = infer_and_update_geometry(
                pts_fused=geom_points,
                prev_geom=track.current_geom,
                min_points=min_geom_points,
            )
        except Exception as e:
            print(f"[DEBUG] infer_and_update_geometry failed for gid={track.global_id}: {e}")
            return

        track.geom_history.append({
            "obs_count": int(track.obs_count),
            "event": "infer",
            "candidate_shape": candidate_geom.shape,
            "candidate_confidence": candidate_geom.confidence,
            "candidate_params": candidate_geom.params,
            "current_shape": track.current_geom.shape if track.current_geom is not None else "none",
            "current_confidence": track.current_geom.confidence if track.current_geom is not None else "none",
            "num_geom_points": int(geom_points.shape[0]),
        })

        # 候选和当前形状一样：只有新置信度不更差时才更新
        if candidate_geom.shape == track.current_geom.shape:
            if confidence_rank(candidate_geom.confidence) >= confidence_rank(track.current_geom.confidence):
                track.current_geom = candidate_geom

            track.pending_geom = None
            track.pending_count = 0
            return

        # 低置信度新形状，不进入 pending
        if confidence_rank(candidate_geom.confidence) < confidence_rank(self.state_cfg.geometry_min_confidence_to_switch):
            return

        # 候选形状不同：做多帧确认
        if track.pending_geom is None or track.pending_geom.shape != candidate_geom.shape:
            track.pending_geom = candidate_geom
            track.pending_count = 1
        else:
            track.pending_geom = candidate_geom
            track.pending_count += 1

        track.geom_history.append({
            "obs_count": int(track.obs_count),
            "event": "pending_update",
            "pending_shape": track.pending_geom.shape if track.pending_geom is not None else "none",
            "pending_confidence": track.pending_geom.confidence if track.pending_geom is not None else "none",
            "pending_count": int(track.pending_count),
        })

        if track.pending_count >= self.state_cfg.geometry_confirm_frames:
            track.current_geom = track.pending_geom
            track.pending_geom = None
            track.pending_count = 0

            track.geom_history.append({
                "obs_count": int(track.obs_count),
                "event": "switch_confirmed",
                "new_current_shape": track.current_geom.shape if track.current_geom is not None else "none",
                "new_current_confidence": track.current_geom.confidence if track.current_geom is not None else "none",
            })

    def _accumulate_into_track(
        self,
        track: ObjectTrack,
        new_points: np.ndarray,
        new_colors: np.ndarray,
        keyframe_index: int,
        local_id: int,
        frame_idx: int,
        info: Optional[Dict[str, Any]] = None,
        kf_dir: Optional[str] = None,
    ):
        new_points, new_colors = denoise_points_and_colors(
            new_points,
            new_colors,
            voxel_size=self.state_cfg.observation_voxel_size,
            nb_neighbors=self.state_cfg.observation_nb_neighbors,
            std_ratio=self.state_cfg.observation_std_ratio,
            keep_largest_cluster=False,
        )

        if new_points.shape[0] < self.state_cfg.min_points_per_object:
            return

        if track.accum_points_ref.shape[0] == 0:
            track.accum_points_ref = new_points.astype(np.float32)
            track.accum_colors_ref = new_colors.astype(np.float32)
        else:
            track.accum_points_ref = np.concatenate(
                [track.accum_points_ref, new_points.astype(np.float32)],
                axis=0
            )
            track.accum_colors_ref = np.concatenate(
                [track.accum_colors_ref, new_colors.astype(np.float32)],
                axis=0
            )

        track.obs_count += 1
        track.last_seen_keyframe = int(keyframe_index)
        track.center_ref = compute_pointcloud_center(track.accum_points_ref)
        track.extent_ref = compute_pointcloud_extent(track.accum_points_ref)
        track.mean_color = mean_color_safe(track.accum_colors_ref)

        track.members.append({
            "keyframe_index": int(keyframe_index),
            "frame_idx": int(frame_idx),
            "local_id": int(local_id),
            "num_points_used": int(new_points.shape[0]),
        })

        self._update_geometry_for_track(
            track=track,
            info=info,
            kf_dir=kf_dir,
        )

    def _save_track_incremental_cloud(
        self,
        track: ObjectTrack,
        keyframe_index: int,
    ):
        if track.accum_points_ref.shape[0] == 0:
            return

        latest_path = os.path.join(self.incremental_dir, f"global_object_{track.global_id:02d}_latest.ply")
        save_pcd_points_and_colors(latest_path, track.accum_points_ref, track.accum_colors_ref)

        if self.state_cfg.save_fused_every_keyframe:
            kf_path = os.path.join(
                self.incremental_dir,
                f"global_object_{track.global_id:02d}_kf_{keyframe_index:04d}.ply"
            )
            save_pcd_points_and_colors(kf_path, track.accum_points_ref, track.accum_colors_ref)

    def _save_tracks_to_keyframe_dir(self, kf_dir: str):
        ensure_dir(kf_dir)

        debug_items = []
        for gid in sorted(self.tracks.keys()):
            tr = self.tracks[gid]
            if tr.accum_points_ref.shape[0] == 0:
                continue

            save_path = os.path.join(kf_dir, f"fused_global_object_{gid:02d}.ply")
            save_pcd_points_and_colors(save_path, tr.accum_points_ref, tr.accum_colors_ref)

            debug_items.append({
                "global_id": gid,
                "num_points": int(tr.accum_points_ref.shape[0]),
                "obs_count": int(tr.obs_count),
                "last_seen_keyframe": int(tr.last_seen_keyframe),
                "geom_shape": tr.current_geom.shape if tr.current_geom is not None else "none",
                "geom_confidence": tr.current_geom.confidence if tr.current_geom is not None else "none",
                "pending_geom_shape": tr.pending_geom.shape if tr.pending_geom is not None else "none",
                "pending_count": int(tr.pending_count),
            })

        save_json(os.path.join(kf_dir, "fused_tracks_debug.json"), {
            "allowed_global_ids": self.allowed_global_ids,
            "tracks": debug_items,
        })

    def _save_geometry_meshes_to_keyframe_dir(self, kf_dir: str):
        if not self.state_cfg.geometry_save_mesh_every_keyframe:
            return

        ensure_dir(kf_dir)

        for gid in sorted(self.tracks.keys()):
            tr = self.tracks[gid]
            if tr.accum_points_ref.shape[0] == 0:
                continue

            if tr.current_geom is None:
                continue

            try:
                mesh = self._build_mesh_for_geom(tr.current_geom)
                if mesh is None:
                    continue

                mesh_path = os.path.join(kf_dir, f"geom_global_object_{gid:02d}.ply")
                mesh.export(mesh_path)
            except Exception as e:
                print(f"[DEBUG] mesh export failed for gid={gid}: {e}")

    def _update_one_object_track(
        self,
        frame_obj: Dict[str, Any],
        global_id: int,
        kf_dir: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pts = frame_obj["points_ref"]
        cols = frame_obj["colors"]

        local_id = int(frame_obj["local_id"])
        keyframe_index = int(frame_obj["keyframe_index"])
        frame_idx = int(frame_obj["frame_idx"])

        if pts.shape[0] < self.state_cfg.min_points_per_object:
            return {
                "local_id": local_id,
                "keyframe_index": keyframe_index,
                "global_id": int(global_id),
                "status": "ignored",
                "reason": "too_few_raw_points",
                "num_points": int(pts.shape[0]),
            }

        if global_id not in self.tracks:
            track = self._init_track_from_object(
                frame_obj=frame_obj,
                global_id=global_id,
            )
            self.tracks[global_id] = track
            self._save_track_incremental_cloud(track, keyframe_index)

            return {
                "local_id": local_id,
                "keyframe_index": keyframe_index,
                "global_id": int(global_id),
                "status": "initialized",
                "num_points_accum": int(track.accum_points_ref.shape[0]),
                "geom_shape": track.current_geom.shape if track.current_geom is not None else "none",
                "geom_confidence": track.current_geom.confidence if track.current_geom is not None else "none",
            }

        track = self.tracks[global_id]
        self._accumulate_into_track(
            track=track,
            new_points=pts,
            new_colors=cols,
            keyframe_index=keyframe_index,
            local_id=local_id,
            frame_idx=frame_idx,
            info=info,
            kf_dir=kf_dir,
        )
        self._save_track_incremental_cloud(track, keyframe_index)

        return {
            "local_id": local_id,
            "keyframe_index": keyframe_index,
            "global_id": int(global_id),
            "status": "updated",
            "num_points_accum": int(track.accum_points_ref.shape[0]),
            "geom_shape": track.current_geom.shape if track.current_geom is not None else "none",
            "geom_confidence": track.current_geom.confidence if track.current_geom is not None else "none",
            "pending_geom_shape": track.pending_geom.shape if track.pending_geom is not None else "none",
            "pending_count": int(track.pending_count),
        }

    def _save_global_map(self):
        save_json(os.path.join(self.save_root, "global_id_map_incremental.json"), {
            "status": "ok",
            "allowed_global_ids": self.allowed_global_ids,
            "num_global_objects": len(self.allowed_global_ids),
            "keyframes": self.per_keyframe_global_map,
        })