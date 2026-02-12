"""
Hesitation Detection Configuration
망설임 감지 모델 설정
"""
import os
from pathlib import Path

# 기본 경로 설정
# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DAISEE_DIR = DATA_DIR / "daisee"  # hesitationLearning/data/daisee/
MODEL_DIR = BASE_DIR / "models"

# 캐시 경로 설정
CACHE_DIR = DATA_DIR / "cache"
FEATURES_CACHE_FILE = CACHE_DIR / "features_cache.npz"
LABELS_CACHE_FILE = CACHE_DIR / "labels_cache.json"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 모델 저장 경로
MODEL_PATH = MODEL_DIR / "hesitation_model.joblib"
SCALER_PATH = MODEL_DIR / "hesitation_scaler.joblib"

# MediaPipe 설정
MEDIAPIPE_CONFIG = {
    "max_num_faces": 1,
    "refine_landmarks": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# 얼굴 랜드마크 인덱스 (망설임 관련 주요 포인트)
# MediaPipe Face Mesh: 468 landmarks
LANDMARK_INDICES = {
    # 눈썹 (혼란/망설임 시 찌푸림)
    "left_eyebrow": [70, 63, 105, 66, 107],
    "right_eyebrow": [336, 296, 334, 293, 300],
    # 눈 (깜빡임/시선)
    "left_eye": [33, 160, 158, 133, 153, 144],
    "right_eye": [362, 385, 387, 263, 373, 380],
    # 입 (입술 움직임)
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308],
    # 얼굴 윤곽 (머리 기울기)
    "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
}

# DAiSEE 라벨 설정
AFFECTIVE_STATES = ["Boredom", "Engagement", "Confusion", "Frustration"]
TARGET_STATE = "Confusion"  # 망설임 = Confusion

# 분류 설정
NUM_CLASSES = 4  # Very Low(0), Low(1), High(2), Very High(3)
BINARY_THRESHOLD = 2  # >= 2 이면 망설임으로 분류 (이진 분류 시)

# 특징 추출 설정
FEATURE_CONFIG = {
    "sample_rate": 5,  # 비디오에서 N 프레임마다 샘플링
    "aggregation": ["mean", "std", "min", "max"],  # 시계열 통계
}

# 학습 설정
TRAIN_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "max_depth": 10,
}

# LSTM 모델 설정 (시계열)
MAX_SEQ_LEN = 150  # 최대 시퀀스 길이 (프레임 수, 약 5~10초) - Padding/Truncating 기준
INPUT_SIZE = 13    # 입력 특징 차원 (get_feature_names() 개수)
BATCH_SIZE = 32
