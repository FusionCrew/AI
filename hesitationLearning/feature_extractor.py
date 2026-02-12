"""
Feature Extractor using MediaPipe Face Mesh (Tasks API)
MediaPipe를 사용한 얼굴 랜드마크 기반 특징 추출
"""
import cv2
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path

import os

# MediaPipe/TensorFlow 불필요한 로그 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import MEDIAPIPE_CONFIG, LANDMARK_INDICES, FEATURE_CONFIG, BASE_DIR


# 모델 파일 경로
MODEL_ASSET_PATH = BASE_DIR / "models" / "face_landmarker.task"


def download_model_if_needed():
    """MediaPipe Face Landmarker 모델 다운로드"""
    import urllib.request
    
    MODEL_ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not MODEL_ASSET_PATH.exists():
        print("Downloading Face Landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, str(MODEL_ASSET_PATH))
        print(f"Model downloaded to {MODEL_ASSET_PATH}")


class FaceMeshExtractor:
    """MediaPipe Face Landmarker를 사용한 얼굴 랜드마크 추출기"""
    
    def __init__(self):
        download_model_if_needed()
        
        base_options = python.BaseOptions(model_asset_path=str(MODEL_ASSET_PATH))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=MEDIAPIPE_CONFIG["max_num_faces"],
            min_face_detection_confidence=MEDIAPIPE_CONFIG["min_detection_confidence"],
            min_face_presence_confidence=MEDIAPIPE_CONFIG["min_tracking_confidence"],
            min_tracking_confidence=MEDIAPIPE_CONFIG["min_tracking_confidence"],
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        
        # 시각화용
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.results = None # 최근 감지 결과 저장
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        이미지에서 얼굴 랜드마크 추출
        
        Args:
            image: BGR 이미지 (OpenCV 형식)
            
        Returns:
            landmarks: (478, 3) 형태의 랜드마크 좌표 또는 None
        """
        # BGR -> RGB 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        
        results = self.landmarker.detect(mp_image)
        self.results = results # 결과 저장 (시각화용)
        
        if not results.face_landmarks:
            return None
        
        # 첫 번째 얼굴의 랜드마크 추출
        face_landmarks = results.face_landmarks[0]
        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks
        ])
        
        return landmarks
    
    def compute_hesitation_features(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        랜드마크에서 망설임 관련 특징 계산
        
        Features:
        - 눈썹 위치/각도 (찌푸림 정도)
        - 눈 개폐 정도 (EAR - Eye Aspect Ratio)
        - 입 개폐 정도 (MAR - Mouth Aspect Ratio)
        - 머리 기울기 (roll, pitch, yaw 추정)
        """
        features = {}
        
        # 1. 눈썹 특징 (찌푸림)
        left_brow = landmarks[LANDMARK_INDICES["left_eyebrow"]]
        right_brow = landmarks[LANDMARK_INDICES["right_eyebrow"]]
        
        # 눈썹 중심 Y 위치 (낮을수록 찌푸림)
        features["left_brow_height"] = np.mean(left_brow[:, 1])
        features["right_brow_height"] = np.mean(right_brow[:, 1])
        features["brow_height_diff"] = abs(features["left_brow_height"] - features["right_brow_height"])
        
        # 눈썹 기울기
        features["left_brow_slope"] = self._compute_slope(left_brow)
        features["right_brow_slope"] = self._compute_slope(right_brow)
        
        # 2. 눈 특징 (EAR)
        left_eye = landmarks[LANDMARK_INDICES["left_eye"]]
        right_eye = landmarks[LANDMARK_INDICES["right_eye"]]
        
        features["left_ear"] = self._compute_ear(left_eye)
        features["right_ear"] = self._compute_ear(right_eye)
        features["avg_ear"] = (features["left_ear"] + features["right_ear"]) / 2
        
        # 3. 입 특징 (MAR)
        mouth = landmarks[LANDMARK_INDICES["mouth"]]
        features["mar"] = self._compute_mar(mouth)
        
        # 4. 머리 기울기 추정
        face_oval = landmarks[LANDMARK_INDICES["face_oval"]]
        features["head_tilt"] = self._estimate_head_tilt(face_oval)
        
        # 5. 얼굴 중심점 (상대적 위치)
        features["face_center_x"] = np.mean(landmarks[:, 0])
        features["face_center_y"] = np.mean(landmarks[:, 1])
        
        # 6. 얼굴 대칭성 (비대칭일수록 혼란)
        features["face_asymmetry"] = self._compute_asymmetry(landmarks)
        
        return features
    
    def _compute_slope(self, points: np.ndarray) -> float:
        """점들의 기울기 계산"""
        if len(points) < 2:
            return 0.0
        x = points[:, 0]
        y = points[:, 1]
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    def _compute_ear(self, eye: np.ndarray) -> float:
        """Eye Aspect Ratio 계산"""
        # 수직 거리
        v1 = np.linalg.norm(eye[1] - eye[5])
        v2 = np.linalg.norm(eye[2] - eye[4])
        # 수평 거리
        h = np.linalg.norm(eye[0] - eye[3])
        
        if h == 0:
            return 0.0
        ear = (v1 + v2) / (2.0 * h)
        return float(ear)
    
    def _compute_mar(self, mouth: np.ndarray) -> float:
        """Mouth Aspect Ratio 계산"""
        # 수직 거리 (입 열림 정도)
        v1 = np.linalg.norm(mouth[2] - mouth[10])
        v2 = np.linalg.norm(mouth[4] - mouth[8])
        # 수평 거리
        h = np.linalg.norm(mouth[0] - mouth[6])
        
        if h == 0:
            return 0.0
        mar = (v1 + v2) / (2.0 * h)
        return float(mar)
    
    def _estimate_head_tilt(self, face_oval: np.ndarray) -> float:
        """머리 기울기 추정 (좌우 비대칭)"""
        left_side = face_oval[:len(face_oval)//2]
        right_side = face_oval[len(face_oval)//2:]
        
        left_center_y = np.mean(left_side[:, 1])
        right_center_y = np.mean(right_side[:, 1])
        
        return float(left_center_y - right_center_y)
    
    def _compute_asymmetry(self, landmarks: np.ndarray) -> float:
        """얼굴 비대칭 정도 계산"""
        center_x = np.mean(landmarks[:, 0])
        
        left_mask = landmarks[:, 0] < center_x
        right_mask = landmarks[:, 0] >= center_x
        
        left_points = landmarks[left_mask]
        right_points = landmarks[right_mask]
        
        if len(left_points) == 0 or len(right_points) == 0:
            return 0.0
        
        # 좌우 대칭점들의 Y 차이
        left_y_std = np.std(left_points[:, 1])
        right_y_std = np.std(right_points[:, 1])
        
        return float(abs(left_y_std - right_y_std))
    
    def close(self):
        """리소스 해제"""
        pass  # FaceLandmarker는 자동 정리됨


def extract_features_from_image(image: np.ndarray) -> Optional[Dict[str, float]]:
    """단일 이미지에서 특징 추출"""
    extractor = FaceMeshExtractor()
    try:
        landmarks = extractor.extract_landmarks(image)
        if landmarks is None:
            return None
        features = extractor.compute_hesitation_features(landmarks)
        return features
    finally:
        extractor.close()


def extract_features_from_video(
    video_path: str,
    sample_rate: int = FEATURE_CONFIG["sample_rate"]
) -> Optional[np.ndarray]:
    """
    비디오에서 프레임별 특징 추출 후 통계 집계
    
    Args:
        video_path: 비디오 파일 경로
        sample_rate: N 프레임마다 샘플링
        
    Returns:
        aggregated_features: 집계된 특징 벡터 또는 None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    extractor = FaceMeshExtractor()
    all_features = []
    frame_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                landmarks = extractor.extract_landmarks(frame)
                if landmarks is not None:
                    features = extractor.compute_hesitation_features(landmarks)
                    all_features.append(list(features.values()))
            
            frame_idx += 1
    finally:
        cap.release()
        extractor.close()
    
    if not all_features:
        return None
    
    # 시계열 통계 집계
    return np.array(all_features)


def get_feature_names() -> List[str]:
    """특징 이름 목록 반환"""
    return [
        "left_brow_height", "right_brow_height", "brow_height_diff",
        "left_brow_slope", "right_brow_slope",
        "left_ear", "right_ear", "avg_ear",
        "mar", "head_tilt",
        "face_center_x", "face_center_y", "face_asymmetry"
    ]
