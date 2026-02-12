"""
Feature Extraction and Caching
특징 추출 및 캐싱 - 모든 비디오에서 특징을 미리 추출하여 저장
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

from .config import DAISEE_DIR, DATA_DIR, TARGET_STATE, BINARY_THRESHOLD
from .feature_extractor import extract_features_from_video, get_feature_names


# 캐시 저장 경로
CACHE_DIR = DATA_DIR / "cache"
FEATURES_CACHE_FILE = CACHE_DIR / "features_cache.npz"
LABELS_CACHE_FILE = CACHE_DIR / "labels_cache.json"


def extract_all_features(max_samples_per_split: int = None, force: bool = False):
    """
    모든 데이터셋에서 특징을 추출하고 캐시 파일로 저장
    
    Args:
        max_samples_per_split: 각 split당 최대 샘플 수
        force: True면 기존 캐시 무시하고 재추출
    """
    print("=" * 60)
    print("Feature Extraction and Caching")
    print("=" * 60)
    
    # 기존 캐시 확인
    if not force and FEATURES_CACHE_FILE.exists() and LABELS_CACHE_FILE.exists():
        print("\n[INFO] Cache files already exist!")
        print(f"  Features: {FEATURES_CACHE_FILE}")
        print(f"  Labels: {LABELS_CACHE_FILE}")
        print("\n  Use --force to re-extract features")
        return
    
    # 캐시 디렉토리 생성
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    all_features = {}
    all_labels = {}
    
    for split in ["Train", "Validation", "Test"]:
        print(f"\n[INFO] Extracting features from {split} split...")
        
        # 라벨 파일 로드
        label_file = DAISEE_DIR / "Labels" / f"{split}Labels.csv"
        if not label_file.exists():
            print(f"  [WARNING] Label file not found: {label_file}")
            continue
        
        df = pd.read_csv(label_file)
        
        if max_samples_per_split:
            df = df.head(max_samples_per_split)
        
        features_list = []
        labels_list = []
        clip_ids = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}"):
            clip_id = row["ClipID"]
            confusion_level = row[TARGET_STATE]
            
            # 비디오 경로 찾기
            video_path = _get_video_path(clip_id, split)
            if video_path is None:
                continue
            
            # 특징 추출
            features = extract_features_from_video(str(video_path))
            if features is None:
                continue
            
            features_list.append(features)
            labels_list.append(confusion_level)
            clip_ids.append(clip_id)
        
        if features_list:
            # 시퀀스 데이터는 길이가 다르므로 object array로 저장
            all_features[split] = np.array(features_list, dtype=object)
            all_labels[split] = {
                "labels": labels_list,
                "clip_ids": clip_ids,
                "count": len(labels_list)
            }
            print(f"  Extracted {len(features_list)} samples")
    
    # 캐시 저장
    print("\n[INFO] Saving cache files...")
    
    # Features 및 Labels 저장 (numpy compressed)
    np.savez_compressed(
        FEATURES_CACHE_FILE,
        train_features=all_features.get("Train", np.array([], dtype=object)),
        train_labels=np.array(all_labels.get("Train", {}).get("labels", [])),
        val_features=all_features.get("Validation", np.array([], dtype=object)),
        val_labels=np.array(all_labels.get("Validation", {}).get("labels", [])),
        test_features=all_features.get("Test", np.array([], dtype=object)),
        test_labels=np.array(all_labels.get("Test", {}).get("labels", []))
    )
    print(f"  Features and labels saved to {FEATURES_CACHE_FILE}")
    
    # Labels 저장 (JSON)
    with open(LABELS_CACHE_FILE, "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"  Labels saved to {LABELS_CACHE_FILE}")
    
    # 통계 출력
    print("\n[INFO] Extraction Summary:")
    for split, info in all_labels.items():
        print(f"  {split}: {info['count']} samples")
    
    print("\n" + "=" * 60)
    print("Feature extraction completed successfully!")
    print("=" * 60)


def load_cached_features(binary: bool = True):
    """
    캐시된 특징 로드
    
    Args:
        binary: True면 이진 분류 라벨로 변환
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    if not FEATURES_CACHE_FILE.exists() or not LABELS_CACHE_FILE.exists():
        raise FileNotFoundError(
            "Cache files not found. Run feature extraction first:\n"
            "  python -m hesitation.extract_features"
        )
    
    print("[INFO] Loading cached features...")
    
    # Features 로드
    data = np.load(FEATURES_CACHE_FILE)
    X_train = data["train"]
    X_val = data["validation"]
    X_test = data["test"]
    
    # Labels 로드
    with open(LABELS_CACHE_FILE, "r") as f:
        labels_data = json.load(f)
    
    y_train = np.array(labels_data["Train"]["labels"])
    y_val = np.array(labels_data["Validation"]["labels"])
    y_test = np.array(labels_data["Test"]["labels"])
    
    # Train + Validation 합치기
    if len(X_val) > 0:
        X_train = np.vstack([X_train, X_val])
        y_train = np.concatenate([y_train, y_val])
    
    # 이진 분류로 변환
    if binary:
        y_train = (y_train >= BINARY_THRESHOLD).astype(int)
        y_test = (y_test >= BINARY_THRESHOLD).astype(int)
    
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, y_train, X_test, y_test


def check_cache_exists() -> bool:
    """캐시 파일 존재 여부 확인"""
    return FEATURES_CACHE_FILE.exists() and LABELS_CACHE_FILE.exists()


def get_cache_info() -> dict:
    """캐시 정보 반환"""
    if not check_cache_exists():
        return {"exists": False}
    
    with open(LABELS_CACHE_FILE, "r") as f:
        labels_data = json.load(f)
    
    return {
        "exists": True,
        "features_file": str(FEATURES_CACHE_FILE),
        "labels_file": str(LABELS_CACHE_FILE),
        "train_count": labels_data.get("Train", {}).get("count", 0),
        "validation_count": labels_data.get("Validation", {}).get("count", 0),
        "test_count": labels_data.get("Test", {}).get("count", 0)
    }


def _get_video_path(clip_id: str, split: str) -> Path:
    """ClipID로 비디오 경로 찾기"""
    clip_name = clip_id.replace(".avi", "").strip()
    user_id = clip_name[:6]
    
    video_path = DAISEE_DIR / "DataSet" / split / user_id / clip_name / clip_id
    
    if video_path.exists():
        return video_path
    
    video_path_mp4 = video_path.with_suffix(".mp4")
    if video_path_mp4.exists():
        return video_path_mp4
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract and cache features from all videos")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per split"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if cache exists"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show cache info and exit"
    )
    
    args = parser.parse_args()
    
    if args.info:
        info = get_cache_info()
        if info["exists"]:
            print("Cache exists:")
            print(f"  Train: {info['train_count']} samples")
            print(f"  Validation: {info['validation_count']} samples")
            print(f"  Test: {info['test_count']} samples")
        else:
            print("No cache found")
        return
    
    extract_all_features(
        max_samples_per_split=args.max_samples,
        force=args.force
    )


if __name__ == "__main__":
    main()
