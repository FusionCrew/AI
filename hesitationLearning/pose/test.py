"""
Body Gesture Detection Test Script
웹캠을 통해 실시간으로 몸짓(팔짱, 손 멈춤 등) 감지를 테스트합니다.
"""
import cv2
import time
import argparse
import sys
import os

# 상위 폴더(프로젝트 루트)를 모듈 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from hesitationLearning.inference import HesitationDetector

def main():
    parser = argparse.ArgumentParser(description="Test Body Gesture Detection")
    parser.add_argument("--src", type=int, default=0, help="Camera source index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (optional)")
    args = parser.parse_args()

    # Detector 초기화
    print("[INFO] Initializing HesitationDetector...")
    detector = HesitationDetector(binary=True)
    
    # 비디오 소스 설정
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(args.src)
        
    if not cap.isOpened():
        print("[ERROR] Cannot open video source")
        return

    print("[INFO] Starting video stream. Press 'q' to exit.")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 감지 수행
        result = detector.detect_from_image(frame)
        
        # 결과 시각화
        body_gestures = result.get("body_gestures", {})
        hesitation_label = result.get("label", "unknown")
        confidence = result.get("confidence", 0.0)
        
        # 1. 기본 정보 표시
        cv2.putText(frame, f"Hesitation: {hesitation_label} ({confidence:.2f})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 2. Body Gesture 표시
        y_pos = 60
        if body_gestures.get("extract_success"):
            for gesture, detected in body_gestures.items():
                if gesture == "extract_success":
                    continue
                    
                color = (0, 0, 255) if detected else (200, 200, 200)
                text = f"{gesture}: {'YES' if detected else 'NO'}"
                cv2.putText(frame, text, (10, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_pos += 30
        else:
            cv2.putText(frame, "Body Pose Not Detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Body Gesture Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()
