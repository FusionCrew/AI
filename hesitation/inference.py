"""
Inference Module for Hesitation Detection
망설임 감지 추론 모듈
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import base64

from .config import MODEL_PATH, SCALER_PATH, NUM_CLASSES, BINARY_THRESHOLD
from .feature_extractor import FaceMeshExtractor, extract_features_from_video
from .model import HesitationClassifier, load_pretrained_model


class HesitationDetector:
    """망설임 감지 추론 클래스"""
    
    def __init__(self, binary: bool = False):
        """
        Args:
            binary: 이진 분류 모드
        """
        self.binary = binary
        self.classifier = None
        self.extractor = None
        self._load_model()
    
    def _load_model(self):
        """모델 및 추출기 초기화"""
        self.classifier = load_pretrained_model(binary=self.binary)
        if self.classifier is None:
            raise RuntimeError(
                "No pretrained model found. Please train the model first using: "
                "python -m hesitation.train"
            )
        self.extractor = FaceMeshExtractor()
    
    def detect_from_image(
        self, 
        image: np.ndarray
    ) -> Dict[str, Union[int, float, str]]:
        """
        단일 이미지에서 망설임 감지
        
        Args:
            image: BGR 이미지 (OpenCV 형식)
            
        Returns:
            result: {
                "hesitation_level": int (0-3 또는 0-1),
                "confidence": float (0-1),
                "label": str ("very_low", "low", "high", "very_high")
            }
        """
        # 랜드마크 추출
        landmarks = self.extractor.extract_landmarks(image)
        
        if landmarks is None:
            return {
                "hesitation_level": -1,
                "confidence": 0.0,
                "label": "no_face_detected",
                "error": "No face detected in the image"
            }
        
        # 특징 추출
        features = self.extractor.compute_hesitation_features(landmarks)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # 특징을 4배 확장 (aggregation과 맞추기 위해)
        # 단일 이미지는 mean=값, std=0, min=값, max=값
        expanded = np.concatenate([
            feature_vector,  # mean
            np.zeros_like(feature_vector),  # std
            feature_vector,  # min
            feature_vector,  # max
        ], axis=1)
        
        # 예측
        prediction = self.classifier.predict(expanded)[0]
        probabilities = self.classifier.predict_proba(expanded)[0]
        confidence = float(np.max(probabilities))
        
        # 라벨 변환
        if self.binary:
            label = "hesitating" if prediction == 1 else "not_hesitating"
        else:
            labels = ["very_low", "low", "high", "very_high"]
            label = labels[prediction] if prediction < len(labels) else "unknown"
        
        return {
            "hesitation_level": int(prediction),
            "confidence": confidence,
            "label": label,
            "probabilities": probabilities.tolist()
        }
    
    def detect_from_base64(self, base64_image: str) -> Dict[str, Union[int, float, str]]:
        """
        Base64 인코딩된 이미지에서 망설임 감지
        
        Args:
            base64_image: Base64 인코딩된 이미지 문자열
            
        Returns:
            result: 감지 결과
        """
        try:
            # Base64 디코딩
            image_data = base64.b64decode(base64_image)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    "hesitation_level": -1,
                    "confidence": 0.0,
                    "label": "decode_error",
                    "error": "Failed to decode image"
                }
            
            return self.detect_from_image(image)
            
        except Exception as e:
            return {
                "hesitation_level": -1,
                "confidence": 0.0,
                "label": "error",
                "error": str(e)
            }
    
    def detect_from_file(self, file_path: str) -> Dict[str, Union[int, float, str]]:
        """
        파일에서 망설임 감지
        
        Args:
            file_path: 이미지 파일 경로
            
        Returns:
            result: 감지 결과
        """
        image = cv2.imread(file_path)
        
        if image is None:
            return {
                "hesitation_level": -1,
                "confidence": 0.0,
                "label": "file_error",
                "error": f"Failed to read image: {file_path}"
            }
        
        return self.detect_from_image(image)
    
    def detect_from_video(
        self, 
        video_path: str
    ) -> Dict[str, Union[int, float, str, list]]:
        """
        비디오에서 망설임 감지 (전체 비디오 분석)
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            result: 감지 결과 (시계열 통계 기반)
        """
        features = extract_features_from_video(video_path)
        
        if features is None:
            return {
                "hesitation_level": -1,
                "confidence": 0.0,
                "label": "video_error",
                "error": "Failed to extract features from video"
            }
        
        feature_vector = features.reshape(1, -1)
        
        prediction = self.classifier.predict(feature_vector)[0]
        probabilities = self.classifier.predict_proba(feature_vector)[0]
        confidence = float(np.max(probabilities))
        
        if self.binary:
            label = "hesitating" if prediction == 1 else "not_hesitating"
        else:
            labels = ["very_low", "low", "high", "very_high"]
            label = labels[prediction] if prediction < len(labels) else "unknown"
        
        return {
            "hesitation_level": int(prediction),
            "confidence": confidence,
            "label": label,
            "probabilities": probabilities.tolist()
        }
    
    def close(self):
        """리소스 해제"""
        if self.extractor:
            self.extractor.close()


# 싱글톤 인스턴스 (FastAPI에서 재사용)
_detector_instance: Optional[HesitationDetector] = None


def get_detector(binary: bool = False) -> HesitationDetector:
    """싱글톤 Detector 인스턴스 반환"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = HesitationDetector(binary=binary)
    return _detector_instance


def detect_hesitation(image: np.ndarray, binary: bool = False) -> Dict:
    """
    간편 함수: 이미지에서 망설임 감지
    
    Args:
        image: BGR 이미지
        binary: 이진 분류 모드
        
    Returns:
        result: 감지 결과
    """
    detector = get_detector(binary=binary)
    return detector.detect_from_image(image)
