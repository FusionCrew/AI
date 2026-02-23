import torch
import numpy as np
import cv2
import mediapipe as mp
from .emotion_model import ResidualEmotionNet
from pathlib import Path

MODEL_PATH = Path("models") / "emotion_resnet_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EmotionDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            # Realtime stream: tracking mode is more robust than per-frame static detection.
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35,
        )

        
        if not MODEL_PATH.exists():
            print(f"[WARNING] Emotion model not found at {MODEL_PATH}")
            self.model = None
            return

        # Load Model
        self.model = ResidualEmotionNet(input_size=1434).to(DEVICE)
        
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
            self.model.eval()
            print(f"Emotion Model loaded on {DEVICE}")
        except Exception as e:
            print(f"[ERROR] Failed to load Emotion Model: {e}")
            self.model = None

    def process_frame(self, image: np.ndarray):
        """
        Process an image (BGR) and return emotion prediction.
        """
        if self.model is None:
            return {"prediction": "Model Error", "anxiety_score": 0.0}

        # MediaPipe FaceMesh
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)

        # Small-face fallback: upsample and retry once.
        if not results.multi_face_landmarks:
            h, w = image.shape[:2]
            up = cv2.resize(image, (max(2 * w, 1), max(2 * h, 1)), interpolation=cv2.INTER_CUBIC)
            up_rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
            results = self.mp_face_mesh.process(up_rgb)

        if not results.multi_face_landmarks:
            return {"prediction": "No Face", "anxiety_score": 0.0}
            
        # Extract Landmarks & Normalize
        landmarks_raw = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks_raw.append([lm.x, lm.y, lm.z])
        
        landmarks_raw = np.array(landmarks_raw)
        
        # Center by nose tip (index 1)
        nose_tip = landmarks_raw[1]
        landmarks_centered = landmarks_raw - nose_tip
        
        # Scale
        max_dist = np.max(np.linalg.norm(landmarks_centered, axis=1))
        if max_dist > 0:
            landmarks_normalized = landmarks_centered / max_dist
        else:
            landmarks_normalized = landmarks_centered
            
        # Flatten
        features = landmarks_normalized.flatten().tolist()
        
        # Inference
        try:
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                
            prob_anxious = probabilities[0][1].item()
            label = "Anxious" if prob_anxious > 0.5 else "Neutral"
            
            return {
                "prediction": label,
                "anxiety_score": round(prob_anxious * 100, 2), # Percentage
                "face_landmarks": results.multi_face_landmarks[0] # Return for visualization if needed
            }
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return {"prediction": "Error", "anxiety_score": 0.0}

    def close(self):
        self.mp_face_mesh.close()
