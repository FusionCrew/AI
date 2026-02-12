import cv2
import argparse
import time
from pathlib import Path
from hesitationLearning.inference import HesitationDetector

def main():
    parser = argparse.ArgumentParser(description="Test Hesitation Detection on Video File")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output_hesitation.mp4", help="Path to output video file")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video file not found: {video_path}")
        return

    print("Initializing Hesitation Detector...")
    detector = HesitationDetector(binary=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return

    # 비디오 정보 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[INFO] Video: {width}x{height}, {fps} FPS, {total_frames} frames")

    # 결과 저장용 Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"[INFO] Processing... Output will be saved to {args.output}")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 추론 실행
        result = detector.detect_from_image(frame, display=True)
        
        # 결과 시각화
        annotated_frame = result.get("annotated_image", frame)
        hesitation_score = result["confidence"]
        is_hesitating = result["label"] == "hesitation"
        
        # 정보 출력 (상단)
        # 1. Progress
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 2. Hesitation Score (Bar Graph & Text)
        bar_x, bar_y, bar_w, bar_h = 10, 60, 200, 20
        cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        
        fill_w = int(bar_w * hesitation_score)
        color = (0, 0, 255) if is_hesitating else (0, 255, 0)
        cv2.rectangle(annotated_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        
        status_text = "HESITATION" if is_hesitating else "NORMAL"
        cv2.putText(annotated_frame, f"{status_text} ({hesitation_score:.2f})", (bar_x, bar_y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 3. Body Gestures
        body_gestures = result.get("body_gestures", {})
        y_offset = 120
        for gesture, detected in body_gestures.items():
            if gesture == "extract_success" or not detected:
                continue
            
            gesture_text = f"Body: {gesture.replace('_', ' ').upper()}"
            cv2.putText(annotated_frame, gesture_text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
            y_offset += 30

        # 저장 및 화면 표시 (선택)
        out.write(annotated_frame)
        
        # 진행률 표시 (console)
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            curr_fps = frame_count / elapsed
            print(f"Processing... {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) - {curr_fps:.1f} FPS", end="\r")
            
    print("\n[INFO] Done!")
    cap.release()
    out.release()
    detector.close()
    print(f"[INFO] Saved result to {args.output}")

if __name__ == "__main__":
    main()
