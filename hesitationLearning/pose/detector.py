"""
Pose hesitation features (Euclidean-rule based, explainable).

pip install opencv-python mediapipe numpy
"""
from collections import deque
from typing import Dict, Tuple

import mediapipe as mp
import numpy as np


class BodyGestureDetector:
    # MediaPipe Pose indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24

    DEFAULT_CONFIG = {
        "tau_up": 0.05,
        "d_min": 0.35,
        "d_max": 1.25,
        "eps_speed": 0.03,
        "N": 6,
        "lean_a": 0.15,
        "lean_b": 0.45,
        "dy_a": -0.25,
        "dy_b": -0.10,
        "M": 20,
        "s_a": 0.10,
        "s_b": 0.35,
        "w_hand_hover": 0.5,
        "w_torso_lean": 0.3,
        "w_sway": 0.2,
        "min_visibility": 0.5,
    }

    def __init__(self, config: Dict = None):
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.left_wrist_history = deque(maxlen=int(self.config["N"]))
        self.right_wrist_history = deque(maxlen=int(self.config["N"]))
        self.hip_x_history = deque(maxlen=int(self.config["M"]))
        self.pose_landmarks = None

    def _dist(self, p: Tuple[float, float], q: Tuple[float, float]) -> float:
        return float(((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5)

    def _clamp(self, x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return float(max(lo, min(hi, x)))

    def _pt(self, lm) -> Tuple[float, float]:
        return (float(lm.x), float(lm.y))

    def _hand_hover_score(
        self,
        wrist: Tuple[float, float],
        wrist_history: deque,
        hip_center: Tuple[float, float],
        torso_center: Tuple[float, float],
        shoulder_width: float,
    ) -> float:
        tau_up = float(self.config["tau_up"])
        d_min = float(self.config["d_min"])
        d_max = float(self.config["d_max"])
        eps_speed = float(self.config["eps_speed"])

        cond_up = wrist[1] < (hip_center[1] - tau_up)
        d_norm = self._dist(wrist, torso_center) / shoulder_width
        cond_mid_air = d_min <= d_norm <= d_max
        if not (cond_up and cond_mid_air):
            return 0.0

        if len(wrist_history) < 2:
            return 0.0

        speeds = []
        prev = wrist_history[0]
        for curr in list(wrist_history)[1:]:
            speed = self._dist(curr, prev) / shoulder_width
            speeds.append(speed)
            prev = curr

        if not speeds:
            return 0.0

        still_ratio = sum(1 for s in speeds if s <= eps_speed) / len(speeds)
        return self._clamp(still_ratio)

    def process_frame(self, image_bgr: np.ndarray) -> Dict[str, float]:
        # MediaPipe expects RGB
        results = self.pose.process(image_bgr[:, :, ::-1])

        base = {
            "extract_success": False,
            "pose_score": 0.0,
            "hand_hover": 0.0,
            "torso_lean": 0.0,
            "sway": 0.0,
            "left_hand_hover": 0.0,
            "right_hand_hover": 0.0,
            "shoulder_width": 0.0,
            "invalid_scale": False,
            "pose_points": [],
            "pose_connections": [],
        }
        self.pose_landmarks = None

        if not results.pose_landmarks:
            return base

        lms = results.pose_landmarks.landmark
        idx = [
            self.LEFT_SHOULDER,
            self.RIGHT_SHOULDER,
            self.LEFT_WRIST,
            self.RIGHT_WRIST,
            self.LEFT_HIP,
            self.RIGHT_HIP,
        ]
        min_vis = float(self.config["min_visibility"])
        if any(float(lms[i].visibility) < min_vis for i in idx):
            return base

        self.pose_landmarks = results.pose_landmarks
        base["extract_success"] = True
        base["pose_points"] = [
            {"x": float(lm.x), "y": float(lm.y), "v": float(lm.visibility)}
            for lm in lms
        ]
        base["pose_connections"] = [list(pair) for pair in self.mp_pose.POSE_CONNECTIONS]

        l_sh = self._pt(lms[self.LEFT_SHOULDER])
        r_sh = self._pt(lms[self.RIGHT_SHOULDER])
        l_wr = self._pt(lms[self.LEFT_WRIST])
        r_wr = self._pt(lms[self.RIGHT_WRIST])
        l_hp = self._pt(lms[self.LEFT_HIP])
        r_hp = self._pt(lms[self.RIGHT_HIP])

        shoulder_width = self._dist(l_sh, r_sh)
        base["shoulder_width"] = shoulder_width
        if shoulder_width < 1e-4:
            base["invalid_scale"] = True
            return base

        shoulder_center = ((l_sh[0] + r_sh[0]) * 0.5, (l_sh[1] + r_sh[1]) * 0.5)
        hip_center = ((l_hp[0] + r_hp[0]) * 0.5, (l_hp[1] + r_hp[1]) * 0.5)
        torso_center = (
            (shoulder_center[0] + hip_center[0]) * 0.5,
            (shoulder_center[1] + hip_center[1]) * 0.5,
        )

        # Update histories
        self.left_wrist_history.append(l_wr)
        self.right_wrist_history.append(r_wr)
        self.hip_x_history.append(hip_center[0])

        # 1) hand_hover
        left_hover = self._hand_hover_score(
            l_wr, self.left_wrist_history, hip_center, torso_center, shoulder_width
        )
        right_hover = self._hand_hover_score(
            r_wr, self.right_wrist_history, hip_center, torso_center, shoulder_width
        )
        hand_hover = max(left_hover, right_hover)

        # 2) torso_lean
        vx = shoulder_center[0] - hip_center[0]
        vy = shoulder_center[1] - hip_center[1]
        lean_ratio = abs(vx) / (abs(vy) + 1e-6)
        lean_a = float(self.config["lean_a"])
        lean_b = float(self.config["lean_b"])
        lean_geom = self._clamp((lean_ratio - lean_a) / (lean_b - lean_a))

        dy = shoulder_center[1] - hip_center[1]
        dy_a = float(self.config["dy_a"])
        dy_b = float(self.config["dy_b"])
        dy_score = self._clamp((dy - dy_a) / (dy_b - dy_a))
        torso_lean = self._clamp(0.7 * lean_geom + 0.3 * dy_score)

        # 3) sway
        if len(self.hip_x_history) >= 2:
            range_x = (max(self.hip_x_history) - min(self.hip_x_history)) / shoulder_width
            s_a = float(self.config["s_a"])
            s_b = float(self.config["s_b"])
            sway = self._clamp((range_x - s_a) / (s_b - s_a))
        else:
            sway = 0.0

        pose_score = self._clamp(
            float(self.config["w_hand_hover"]) * hand_hover
            + float(self.config["w_torso_lean"]) * torso_lean
            + float(self.config["w_sway"]) * sway
        )

        base.update(
            {
                "pose_score": pose_score,
                "hand_hover": hand_hover,
                "torso_lean": torso_lean,
                "sway": sway,
                "left_hand_hover": left_hover,
                "right_hand_hover": right_hover,
            }
        )
        return base

    def close(self):
        self.pose.close()
