"""
DAiSEE Dataset Loader
DAiSEE 데이터셋 로더 및 전처리
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm

from .config import (
    DAISEE_DIR, TARGET_STATE, NUM_CLASSES, 
    BINARY_THRESHOLD, FEATURE_CONFIG
)
from .feature_extractor import extract_features_from_video, get_feature_names


class DAiSEEDataset:
    """DAiSEE 데이터셋 로더"""
    
    def __init__(self, data_dir: Path = DAISEE_DIR, binary: bool = False):
        """
        Args:
            data_dir: DAiSEE 데이터셋 경로
            binary: True면 이진 분류 (망설임 있음/없음)
        """
        self.data_dir = Path(data_dir)
        self.binary = binary
        self.labels_dir = self.data_dir / "Labels"
        
    def _load_labels(self, split: str) -> pd.DataFrame:
        """
        라벨 CSV 로드
        
        Args:
            split: "Train", "Validation", or "Test"
        """
        label_file = self.labels_dir / f"{split}Labels.csv"
        if not label_file.exists():
            raise FileNotFoundError(f"Label file not found: {label_file}")
        
        df = pd.read_csv(label_file)
        # 필요한 컬럼만 추출
        # DAiSEE CSV 형식: ClipID, Boredom, Engagement, Confusion, Frustration
        return df
    
    def _get_video_path(self, clip_id: str, split: str) -> Optional[Path]:
        """ClipID로 비디오 경로 찾기"""
        # DAiSEE 구조: DataSet/{split}/{user_id}/{clip_folder}/{clip_id}
        # clip_id 형식: 1100011002.avi
        # user_id: 처음 6자리 (110001)
        # clip_folder: .avi 제외한 clip_id (1100011002)
        
        # .avi 확장자 제거
        clip_name = clip_id.replace(".avi", "").strip()
        user_id = clip_name[:6]  # 처음 6자리가 user_id
        
        # 실제 경로: DataSet/Train/110001/1100011002/1100011002.avi
        video_path = self.data_dir / "DataSet" / split / user_id / clip_name / clip_id
        
        if video_path.exists():
            return video_path
        
        # .mp4 확장자도 확인
        video_path_mp4 = video_path.with_suffix(".mp4")
        if video_path_mp4.exists():
            return video_path_mp4
        
        return None
    
    def load_split(
        self, 
        split: str = "Train",
        max_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        데이터셋 split 로드 및 특징 추출
        
        Args:
            split: "Train", "Validation", or "Test"
            max_samples: 최대 샘플 수 (테스트용)
            
        Returns:
            X: 특징 배열 (n_samples, n_features)
            y: 라벨 배열 (n_samples,)
        """
        df = self._load_labels(split)
        
        if max_samples:
            df = df.head(max_samples)
        
        X_list = []
        y_list = []
        
        print(f"Loading {split} split...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            clip_id = row["ClipID"]
            confusion_level = row[TARGET_STATE]
            
            video_path = self._get_video_path(clip_id, split)
            if video_path is None:
                continue
            
            features = extract_features_from_video(str(video_path))
            if features is None:
                continue
            
            X_list.append(features)
            
            if self.binary:
                # 이진 분류: confusion >= threshold -> 1 (망설임)
                label = 1 if confusion_level >= BINARY_THRESHOLD else 0
            else:
                label = confusion_level
            
            y_list.append(label)
        
        if not X_list:
            raise ValueError(f"No valid samples found in {split} split")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"Loaded {len(X)} samples from {split}")
        return X, y
    
    def load_all(
        self, 
        max_samples_per_split: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        전체 데이터셋 로드
        
        Returns:
            X_train, y_train, X_test, y_test
        """
        X_train, y_train = self.load_split("Train", max_samples_per_split)
        X_val, y_val = self.load_split("Validation", max_samples_per_split)
        X_test, y_test = self.load_split("Test", max_samples_per_split)
        
        # Train + Validation 합치기
        X_train = np.vstack([X_train, X_val])
        y_train = np.concatenate([y_train, y_val])
        
        return X_train, y_train, X_test, y_test


def check_dataset_exists() -> bool:
    """데이터셋 존재 여부 확인"""
    required_paths = [
        DAISEE_DIR / "Labels" / "TrainLabels.csv",
        DAISEE_DIR / "DataSet",
    ]
    
    for path in required_paths:
        if not path.exists():
            print(f"Missing: {path}")
            return False
    
    return True


def get_dataset_stats() -> dict:
    """데이터셋 통계 정보"""
    stats = {}
    
    for split in ["Train", "Validation", "Test"]:
        label_file = DAISEE_DIR / "Labels" / f"{split}Labels.csv"
        if label_file.exists():
            df = pd.read_csv(label_file)
            stats[split] = {
                "total": len(df),
                "confusion_distribution": df[TARGET_STATE].value_counts().to_dict()
            }
    
    return stats
