"""
Inference module for frame-based hesitation detection.

- final_raw = 0.7 * face_score + 0.3 * pose_score
- final_ema = alpha * final_raw + (1 - alpha) * prev_ema
- is_hesitating = final_ema >= threshold

Dependencies:
  pip install opencv-python mediapipe numpy
"""
import base64
import time
from typing import Dict, Optional, Union

import cv2
import numpy as np

from .emotion_extractor import EmotionDetector
from .pose import BodyGestureDetector


DEFAULT_CONFIG = {
    "face_weight": 0.7,
    "pose_weight": 0.3,
    "ema_alpha": 0.2,
    "hesitation_threshold": 0.6,
    "face_override_enabled": True,
    "face_override_threshold": 0.9,
    "face_override_pose_max": 0.05,
}


def get_face_score(frame_bgr: np.ndarray) -> float:
    """TODO: integrate external face model. Must return 0.0~1.0."""
    _ = frame_bgr
    return 0.0


class HesitationDetector:
    """Combined detector with explainable pose features + face score."""

    def __init__(self, binary: bool = False, config: Dict = None, use_face_stub: bool = False):
        self.binary = binary
        self.use_face_stub = use_face_stub
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        self.body_detector = BodyGestureDetector()
        self.emotion_detector = None if use_face_stub else EmotionDetector()
        self.final_ema: Optional[float] = None

    def _clamp(self, x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return float(max(lo, min(hi, x)))

    def _get_face_score(self, frame_bgr: np.ndarray) -> Dict[str, Union[float, str, Dict]]:
        if self.use_face_stub:
            score = self._clamp(get_face_score(frame_bgr))
            return {
                "face_score": score,
                "face_emotion": {
                    "prediction": "TODO",
                    "anxiety_score": round(score * 100.0, 2),
                },
            }

        if self.emotion_detector is None:
            return {
                "face_score": 0.0,
                "face_emotion": {"prediction": "Model Error", "anxiety_score": 0.0},
            }

        face_emotion = self.emotion_detector.process_frame(frame_bgr)
        face_score = self._clamp(float(face_emotion.get("anxiety_score", 0.0)) / 100.0)
        return {"face_score": face_score, "face_emotion": face_emotion}

    def update(self, frame_bgr: np.ndarray, display: bool = False) -> Dict[str, Union[int, float, str, Dict]]:
        result = {
            "hesitation_level": 0,
            "confidence": 0.0,
            "label": "NORMAL",
            "status": "NORMAL",
            "face_score": 0.0,
            "body_score": 0.0,
            "pose_score": 0.0,
            "final_raw": 0.0,
            "final_ema": 0.0,
            "is_hesitating": False,
            "pose_features": {},
            "body_gestures": {},
            "face_emotion": {},
            "decision_reason": "",
        }

        annotated_image = frame_bgr.copy() if display else None

        # 1) Pose explainable features
        pose_result = self.body_detector.process_frame(frame_bgr)
        pose_score = self._clamp(float(pose_result.get("pose_score", 0.0)))
        result["pose_features"] = pose_result
        result["body_gestures"] = pose_result  # backward compatibility
        result["pose_score"] = pose_score
        result["body_score"] = pose_score

        if display and self.body_detector.pose_landmarks:
            self.body_detector.mp_drawing.draw_landmarks(
                annotated_image,
                self.body_detector.pose_landmarks,
                self.body_detector.mp_pose.POSE_CONNECTIONS,
            )

        # 2) Face score (0~1)
        face_result = self._get_face_score(frame_bgr)
        face_score = float(face_result["face_score"])
        result["face_score"] = face_score
        result["face_emotion"] = face_result["face_emotion"]

        # 3) Weighted fusion + EMA
        face_w = float(self.config["face_weight"])
        pose_w = float(self.config["pose_weight"])
        alpha = float(self.config["ema_alpha"])
        threshold = float(self.config["hesitation_threshold"])
        face_override_enabled = bool(self.config["face_override_enabled"])
        face_override_threshold = float(self.config["face_override_threshold"])
        face_override_pose_max = float(self.config["face_override_pose_max"])

        final_raw = self._clamp(face_w * face_score + pose_w * pose_score)
        if self.final_ema is None:
            self.final_ema = final_raw
        else:
            self.final_ema = self._clamp(alpha * final_raw + (1.0 - alpha) * self.final_ema)

        pose_missing = not bool(pose_result.get("extract_success", False))
        face_override = (
            face_override_enabled
            and face_score >= face_override_threshold
            and (pose_missing or pose_score <= face_override_pose_max)
        )

        is_hesitating = (self.final_ema >= threshold) or face_override
        status = "HESITATING" if is_hesitating else "NORMAL"

        result["final_raw"] = round(final_raw, 4)
        result["final_ema"] = round(float(self.final_ema), 4)
        result["confidence"] = round(float(self.final_ema), 4)
        result["is_hesitating"] = is_hesitating
        result["status"] = status
        result["label"] = status
        result["hesitation_level"] = 1 if is_hesitating else 0
        if face_override:
            result["decision_reason"] = "face_override"
        elif self.final_ema >= threshold:
            result["decision_reason"] = "ema_threshold"
        else:
            result["decision_reason"] = "below_threshold"

        if display:
            result["annotated_image"] = annotated_image

        return result

    def detect_from_image(self, image: np.ndarray, display: bool = False) -> Dict[str, Union[int, float, str, Dict]]:
        return self.update(image, display=display)

    def detect_from_file(self, file_path: str) -> Dict:
        image = cv2.imread(file_path)
        if image is None:
            return {"error": f"Failed to read image: {file_path}"}
        return self.update(image)

    def detect_from_base64(self, base64_image: str) -> Dict:
        try:
            image_data = base64.b64decode(base64_image)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return {"error": "Failed to decode image"}
            return self.update(image)
        except Exception as e:
            return {"error": str(e)}

    def close(self):
        if self.body_detector:
            self.body_detector.close()
        if self.emotion_detector:
            self.emotion_detector.close()


_detector_instance: Optional[HesitationDetector] = None


def get_detector(binary: bool = False) -> HesitationDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = HesitationDetector(binary=binary)
    return _detector_instance


def detect_hesitation(image: np.ndarray, binary: bool = False) -> Dict:
    detector = get_detector(binary=binary)
    return detector.update(image)


def _draw_overlay(frame: np.ndarray, out: Dict, fps: float) -> None:
    pf = out.get("pose_features", {}) or {}
    lines = [
        f"face_score: {out.get('face_score', 0.0):.3f}",
        f"pose_score: {out.get('pose_score', 0.0):.3f}",
        f"  hand_hover: {pf.get('hand_hover', 0.0):.3f}",
        f"  torso_lean: {pf.get('torso_lean', 0.0):.3f}",
        f"  sway: {pf.get('sway', 0.0):.3f}",
        f"final_raw: {out.get('final_raw', 0.0):.3f}",
        f"final_ema: {out.get('final_ema', 0.0):.3f}",
        f"status: {out.get('status', 'NORMAL')}",
        f"FPS: {fps:.1f}",
    ]

    y = 28
    for i, t in enumerate(lines):
        color = (0, 0, 255) if i == 7 and out.get("is_hesitating") else (40, 220, 40)
        cv2.putText(frame, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        y += 24


def main():
    detector = HesitationDetector(use_face_stub=False)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    prev = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            dt = max(now - prev, 1e-6)
            prev = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

            out = detector.update(frame, display=True)
            vis = out.get("annotated_image", frame)
            _draw_overlay(vis, out, fps)

            cv2.imshow("Hesitation Detection", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
