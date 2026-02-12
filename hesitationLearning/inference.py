"""
Inference Module for Hesitation Detection
(Refactored: Pose/Body Gesture ONLY active. Face/LSTM disabled.)
"""
import cv2
import numpy as np
from typing import Dict, Optional, Union, List
import base64

# [FACE/LSTM DISABLED]
# from .config import MODEL_PATH, MODEL_DIR, NUM_CLASSES, BINARY_THRESHOLD, MAX_SEQ_LEN
# from .feature_extractor import FaceMeshExtractor, extract_features_from_video
# from .model import HesitationClassifier, load_pretrained_model

from .pose import BodyGestureDetector


class HesitationDetector:
    """
    Combined Hesitation Detection (Face + Body)
    *Currently running in POSE-ONLY mode*
    """
    
    def __init__(self, binary: bool = False):
        self.binary = binary
        
        # 1. Body Gesture (Active)
        self.body_detector = BodyGestureDetector()
        
        # 2. Face Feature Extractor (Disabled)
        self.extractor = None
        # self.extractor = FaceMeshExtractor()
        
        # 3. LSTM Classifier (Disabled)
        self.classifier = None
        # self.classifier = load_pretrained_model(binary=self.binary)
        
    def detect_from_image(
        self, 
        image: np.ndarray,
        display: bool = False
    ) -> Dict[str, Union[int, float, str, Dict]]:
        """
        Detect hesitation from single image (Body Only)
        """
        result = {
            "hesitation_level": -1,
            "confidence": 0.0,
            "label": "pose_only", # Changed from unknown
            "body_gestures": {}
        }
        
        # Copy for visualization
        annotated_image = image.copy() if display else None

        # 1. Body Gesture Detection
        try:
            body_gestures = self.body_detector.process_frame(image)
            result["body_gestures"] = body_gestures
            
            # Visualization (Body Landmarks)
            # BodyGestureDetector uses mp.solutions.pose which returns compatible Proto landmarks for drawing utils
            if display and self.body_detector.pose_landmarks:
                mp_drawing = self.body_detector.mp_drawing
                mp_pose = self.body_detector.mp_pose
                mp_drawing.draw_landmarks(
                    annotated_image,
                    self.body_detector.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
        except Exception as e:
            print(f"[ERROR] Body gesture detection failed: {e}")
            result["body_gestures"] = {"error": str(e)}
        
        # 2. Face Hesitation Detection (Disabled)
        # ... (Code commented out)
        
        if display:
            result["annotated_image"] = annotated_image
            
        return result

    # [FACE/LSTM DISABLED methods]
    # def _prepare_sequence(self, features: np.ndarray) -> np.ndarray: ...
    # def _convert_to_proto(self, landmarks): ...

    def close(self):
        """Release resources"""
        # if self.extractor:
        #     self.extractor.close()
        if self.body_detector:
            self.body_detector.close()

    # Legacy file/video helpers (Simplified to call detect_from_image)
    def detect_from_file(self, file_path: str) -> Dict:
        image = cv2.imread(file_path)
        if image is None:
            return {"error": f"Failed to read image: {file_path}"}
        return self.detect_from_image(image)

    def detect_from_base64(self, base64_image: str) -> Dict:
        try:
            image_data = base64.b64decode(base64_image)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return {"error": "Failed to decode image"}
            return self.detect_from_image(image)
        except Exception as e:
            return {"error": str(e)}


# Singleton instance
_detector_instance: Optional[HesitationDetector] = None

def get_detector(binary: bool = False) -> HesitationDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = HesitationDetector(binary=binary)
    return _detector_instance

def detect_hesitation(image: np.ndarray, binary: bool = False) -> Dict:
    detector = get_detector(binary=binary)
    return detector.detect_from_image(image)
