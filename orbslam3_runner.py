import numpy as np
import cv2
import time


class ORBSLAM3Runner:
    def __init__(self, orbslam3, vocab_path, settings_path, use_viewer=False):
        self.slam = self._create_system(
            orbslam3, vocab_path, settings_path, use_viewer
        )

    def _create_system(self, orbslam3, vocab, settings, use_viewer):
        # 兼容 enum / int
        sensor_candidates = []
        if hasattr(orbslam3, "Sensor") and hasattr(orbslam3.Sensor, "RGBD"):
            sensor_candidates.append(orbslam3.Sensor.RGBD)
        sensor_candidates += [2, 3]  # RGBD 常见取值

        for s in sensor_candidates:
            try:
                return orbslam3.ORBSLAM3(vocab, settings, s, use_viewer)
            except:
                try:
                    return orbslam3.ORBSLAM3(vocab, settings, s)
                except:
                    pass

        raise RuntimeError("Failed to create ORB-SLAM3 system")

    def track(self, color_bgr, depth_u16, depth_scale, timestamp):
        """
        输入：来自 RealSenseStream.read()
        输出：Twc (4x4) 或 None
        """
        # --- format convert ---
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth_m = depth_u16.astype(np.float32) * depth_scale

        Tcw, is_kf = self.slam.track_rgbd(rgb, depth_m, timestamp)
        if Tcw is None:
            return None, False

        Twc = np.linalg.inv(Tcw)
        return Twc, bool(is_kf)
