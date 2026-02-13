"""
Realtime hesitation test script.
Press q to quit.
"""
import argparse
import os
import sys
import time

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from hesitationLearning.inference import HesitationDetector


def main():
    parser = argparse.ArgumentParser(description="Test hesitation detection")
    parser.add_argument("--src", type=int, default=0, help="Camera source index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    args = parser.parse_args()

    detector = HesitationDetector(use_face_stub=False)
    cap = cv2.VideoCapture(args.video if args.video else args.src)
    if not cap.isOpened():
        print("[ERROR] Cannot open video source")
        return

    prev = time.time()
    fps = 0.0

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
        pf = out.get("pose_features", {}) or {}

        lines = [
            f"status: {out.get('status', 'NORMAL')}",
            f"face_score: {out.get('face_score', 0.0):.3f}",
            f"pose_score: {out.get('pose_score', 0.0):.3f}",
            f" hand_hover: {pf.get('hand_hover', 0.0):.3f}",
            f" torso_lean: {pf.get('torso_lean', 0.0):.3f}",
            f" sway: {pf.get('sway', 0.0):.3f}",
            f"final_raw: {out.get('final_raw', 0.0):.3f}",
            f"final_ema: {out.get('final_ema', 0.0):.3f}",
            f"FPS: {fps:.1f}",
        ]

        y = 28
        for i, t in enumerate(lines):
            color = (0, 0, 255) if i == 0 and out.get("is_hesitating") else (0, 255, 0)
            cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 24

        cv2.imshow("Hesitation Test", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
