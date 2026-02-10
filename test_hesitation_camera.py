import cv2
import time
import numpy as np
from hesitationLearning.inference import HesitationDetector

def main():
    print("Initializing Hesitation Detector...")
    # binary=True로 설정하여 학습된 모델 로드
    detector = HesitationDetector(binary=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Starts webcam inference... Press 'q' to quit.")
    
    # FPS 계산용
    prev_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 좌우 반전 (거울 모드)
        frame = cv2.flip(frame, 1)
        
        # 추론 실행
        # display=True로 설정하여 랜드마크 시각화된 이미지 받음
        result = detector.detect_from_image(frame, display=True)
        
        # 결과 시각화
        annotated_frame = result.get("annotated_image", frame)
        hesitation_score = result["confidence"]
        is_hesitating = result["label"] == "hesitation"
        
        # FPS 표시
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        
        # 화면에 정보 출력
        # 1. FPS
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 2. Hesitation Score (Bar Graph & Text)
        bar_x, bar_y, bar_w, bar_h = 10, 60, 200, 20
        cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        fill_w = int(bar_w * hesitation_score)
        color = (0, 0, 255) if is_hesitating else (0, 255, 0) # Red if hesitating, else Green
        cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        
        status_text = "HESITATION" if is_hesitating else "NORMAL"
        cv2.putText(annotated_frame, f"{status_text} ({hesitation_score:.2f})", (bar_x, bar_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. Body Gestures (if any)
        body_gestures = result.get("body_gestures", {})
        y_offset = 120
        for gesture, detected in body_gestures.items():
            if gesture == "extract_success" or not detected:
                continue
            
            # 감지된 제스처 표시
            gesture_text = f"Body: {gesture.replace('_', ' ').upper()}"
            cv2.putText(annotated_frame, gesture_text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            y_offset += 30

        cv2.imshow("Hesitation Detection Test", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()
