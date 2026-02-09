"""
Training Script for Hesitation Detection Model
망설임 감지 모델 학습 스크립트
"""
import argparse
import json
import numpy as np
from pathlib import Path
from datetime import datetime

from .config import MODEL_DIR, DAISEE_DIR
from .dataset import DAiSEEDataset, check_dataset_exists, get_dataset_stats
from .model import HesitationClassifier
from .feature_extractor import get_feature_names


def train(
    binary: bool = True,  # 이진 분류 기본값
    max_samples: int = 10,
    test_mode: bool = False,
    threshold: float = 0.3
):
    """
    모델 학습 실행
    
    Args:
        binary: 이진 분류 여부
        max_samples: 최대 샘플 수 (테스트용)
        test_mode: 테스트 모드 (소량 데이터로 빠른 테스트)
    """
    print("=" * 60)
    print("Hesitation Detection Model Training")
    print("=" * 60)
    
    # 1. 데이터셋 확인
    if not check_dataset_exists():
        print("\n[ERROR] DAiSEE dataset not found!")
        print(f"Please download the dataset and place it in: {DAISEE_DIR}")
        print("Download link: https://www.kaggle.com/datasets/olgaparfenova/daisee")
        return
    
    # 2. 데이터셋 통계
    print("\n[INFO] Dataset Statistics:")
    stats = get_dataset_stats()
    for split, info in stats.items():
        print(f"  {split}: {info['total']} samples")
        print(f"    Confusion distribution: {info['confusion_distribution']}")
    
    # 3. 데이터 로드
    if test_mode and max_samples is None:
        max_samples = 50  # 테스트 모드 기본값
    
    if max_samples:
        print(f"\n[INFO] Using max {max_samples} samples per split")
    
    # 캐시 사용 확인
    from .extract_features import check_cache_exists, load_cached_features
    
    use_cache = check_cache_exists() and max_samples is None
    
    if use_cache:
        print("\n[INFO] Loading cached features (fast)...")
        try:
            X_train, y_train, X_test, y_test = load_cached_features(binary=binary)
        except Exception as e:
            print(f"  [WARNING] Cache load failed: {e}")
            use_cache = False
    
    if not use_cache:
        print("\n[INFO] Loading dataset from videos (slow)...")
        dataset = DAiSEEDataset(binary=binary)
        
        try:
            X_train, y_train, X_test, y_test = dataset.load_all(
                max_samples_per_split=max_samples
            )
        except Exception as e:
            print(f"\n[ERROR] Failed to load dataset: {e}")
            return
    
    print(f"\n[INFO] Dataset loaded:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # 클래스 분포 확인
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"  Class distribution: {dict(zip(unique, counts))}")
    
    # 4. SMOTE로 클래스 균형 맞추기
    print("\n[INFO] Applying SMOTE to balance classes...")
    try:
        from imblearn.over_sampling import SMOTE
        
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        unique_after, counts_after = np.unique(y_train_balanced, return_counts=True)
        print(f"  Before SMOTE: {X_train.shape[0]} samples")
        print(f"  After SMOTE: {X_train_balanced.shape[0]} samples")
        print(f"  Balanced distribution: {dict(zip(unique_after, counts_after))}")
        
        X_train = X_train_balanced
        y_train = y_train_balanced
    except ImportError:
        print("  [WARNING] imbalanced-learn not installed. Skipping SMOTE.")
    except Exception as e:
        print(f"  [WARNING] SMOTE failed: {e}. Using original data.")
    
    # 5. 모델 학습
    print("\n[INFO] Training model...")
    classifier = HesitationClassifier(binary=binary)
    train_metrics = classifier.fit(X_train, y_train)
    
    print("\n[INFO] Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 5. 테스트 평가
    print(f"\n[INFO] Evaluating on test set (threshold={threshold})...")
    test_results = classifier.evaluate(X_test, y_test, threshold=threshold)
    
    print("\n[INFO] Test Metrics:")
    for key, value in test_results.items():
        if key == "classification_report":
            print(f"\n{value}")
        elif key == "confusion_matrix":
            print(f"  Confusion Matrix: {value}")
        else:
            print(f"  {key}: {value:.4f}")
    
    # 6. 특징 중요도
    feature_names = get_feature_names()
    importances = classifier.get_feature_importance()
    
    print("\n[INFO] Top 10 Important Features:")
    indices = importances.argsort()[::-1][:10]
    for i, idx in enumerate(indices):
        if idx < len(feature_names):
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # 7. 모델 저장
    print("\n[INFO] Saving model...")
    classifier.save()
    
    # 8. 그래프 시각화 (Loss와 Accuracy만)
    print("\n[INFO] Generating visualizations...")
    try:
        import matplotlib.pyplot as plt
        
        # 학습 history 가져오기
        history = classifier.get_training_history()
        has_history = len(history.get("loss", [])) > 0
        
        if has_history:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            loss_curve = history["loss"]
            val_scores = history.get("val_loss", [])
            epochs = range(1, len(loss_curve) + 1)
            
            # 8-1. Loss 곡선
            axes[0].plot(epochs, loss_curve, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
            axes[0].set_title("Training Loss", fontsize=14, fontweight='bold')
            axes[0].set_xlabel("Epoch", fontsize=12)
            axes[0].set_ylabel("Loss", fontsize=12)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 8-2. Accuracy (Validation Score) 곡선
            if val_scores:
                axes[1].plot(epochs, val_scores, 'g-', label='Validation Accuracy', linewidth=2, marker='o', markersize=3)
                axes[1].set_title("Validation Accuracy", fontsize=14, fontweight='bold')
                axes[1].set_xlabel("Epoch", fontsize=12)
                axes[1].set_ylabel("Accuracy", fontsize=12)
                axes[1].set_ylim([0, 1.05])
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            else:
                # Validation score가 없으면 Test 결과 표시
                axes[1].text(0.5, 0.5, f"Test Accuracy: {test_results['accuracy']:.2%}\nTest F1: {test_results['f1']:.2%}",
                            transform=axes[1].transAxes, fontsize=16, 
                            verticalalignment='center', horizontalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
                axes[1].set_title("Test Results", fontsize=14, fontweight='bold')
                axes[1].axis('off')
            
            plt.tight_layout()
            
            # 그래프 저장
            graph_path = MODEL_DIR / "training_results.png"
            plt.savefig(graph_path, dpi=150, bbox_inches="tight")
            print(f"  Graph saved to {graph_path}")
            
            # 그래프 표시
            plt.show()
        else:
            print("  [INFO] No training history available for plotting.")
        
    except ImportError:
        print("  [WARNING] matplotlib/seaborn not installed. Skipping visualization.")
    except Exception as e:
        print(f"  [WARNING] Failed to generate graph: {e}")
    
    # 9. 학습 결과 저장
    results = {
        "timestamp": datetime.now().isoformat(),
        "binary": binary,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_metrics": train_metrics,
        "test_metrics": {k: v for k, v in test_results.items() 
                        if k not in ["classification_report", "confusion_matrix"]},
    }
    
    results_path = MODEL_DIR / "training_results.json"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[INFO] Results saved to {results_path}")
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train hesitation detection model")
    parser.add_argument(
        "--binary", 
        action="store_true",
        default=True,
        help="Use binary classification (default: True)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per split (enter a number, e.g. --max-samples 100)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode with minimal data (50 samples if --max-samples not set)"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Classification threshold for binary classification (default: 0.3)"
    )
    
    args = parser.parse_args()
    train(
        binary=args.binary,
        max_samples=args.max_samples,
        test_mode=args.test_mode,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()
