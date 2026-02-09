"""
Hesitation Detection Model
망설임 감지 분류 모델 - MLP (Deep Learning Alternative)
"""
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)

from .config import MODEL_PATH, SCALER_PATH, MODEL_DIR, TRAIN_CONFIG, NUM_CLASSES


class HesitationClassifier:
    """망설임 감지 분류기 (MLP - Multi-Layer Perceptron)"""
    
    def __init__(self, binary: bool = True):  # 기본값을 True로 변경
        """
        Args:
            binary: True면 이진 분류, False면 4-class 분류
        """
        self.binary = binary
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),  # 3층 신경망
            activation="relu",
            solver="adam",  # Adam 옵티마이저
            alpha=0.001,  # L2 정규화
            batch_size=32,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,  # 조기 종료
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=TRAIN_CONFIG["random_state"],
            verbose=True  # 학습 진행 출력
        )
        self.is_fitted = False
        self.training_history: Dict[str, List[float]] = {
            "loss": [],
            "val_loss": []
        }
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        모델 학습
        
        Args:
            X: 특징 배열 (n_samples, n_features)
            y: 라벨 배열 (n_samples,)
            
        Returns:
            metrics: 학습 데이터에 대한 성능 지표
        """
        # 특징 정규화
        X_scaled = self.scaler.fit_transform(X)
        
        # 모델 학습
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        # 학습 history 저장
        self.training_history["loss"] = list(self.model.loss_curve_)
        if hasattr(self.model, 'validation_scores_'):
            self.training_history["val_loss"] = list(self.model.validation_scores_)
        
        # 학습 데이터 성능
        y_pred = self.model.predict(X_scaled)
        metrics = self._compute_metrics(y, y_pred)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        예측
        
        Args:
            X: 특징 배열
            
        Returns:
            predictions: 예측 라벨
        """
    def predict(self, X: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """
        예측
        
        Args:
            X: 특징 배열
            threshold: 이진 분류 임계값 (기본 0.3)
            
        Returns:
            predictions: 예측 라벨
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        
        if self.binary:
            # 이진 분류: 확률 기반 임계값 적용
            proba = self.model.predict_proba(X_scaled)[:, 1]
            return (proba >= threshold).astype(int)
        else:
            # 다중 분류: 가장 높은 확률의 클래스 선택
            return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        확률 예측
        
        Args:
            X: 특징 배열
            
        Returns:
            probabilities: 클래스별 확률
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.3) -> Dict[str, Any]:
        """
        모델 평가
        
        Args:
            X: 테스트 특징 배열
            y: 테스트 라벨
            threshold: 이진 분류 임계값
            
        Returns:
            results: 평가 결과 (metrics + confusion matrix + report)
        """
        y_pred = self.predict(X, threshold=threshold)
        
        metrics = self._compute_metrics(y, y_pred)
        
        results = {
            **metrics,
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, zero_division=0)
        }
        
        return results
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """성능 지표 계산"""
        average = "binary" if self.binary else "weighted"
        
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        }
    
    def save(self, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
        """모델 저장"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # 디렉토리 생성
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # History 저장
        history_path = MODEL_DIR / "training_history.json"
        import json
        with open(history_path, "w") as f:
            json.dump(self.training_history, f)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load(self, model_path: Path = MODEL_PATH, scaler_path: Path = SCALER_PATH):
        """모델 로드"""
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_fitted = True
        
        print(f"Model loaded from {model_path}")
    
    def get_feature_importance(self) -> np.ndarray:
        """특징 중요도 반환 (MLP의 첫 번째 레이어 가중치 절대값)"""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        # MLP: 첫 번째 hidden layer의 가중치 사용
        weights = np.abs(self.model.coefs_[0])
        return np.mean(weights, axis=1)
    
    def get_training_history(self) -> Dict[str, List[float]]:
        """학습 history 반환"""
        return self.training_history


def load_pretrained_model(binary: bool = True) -> Optional[HesitationClassifier]:
    """사전 학습된 모델 로드"""
    classifier = HesitationClassifier(binary=binary)
    
    try:
        classifier.load()
        return classifier
    except FileNotFoundError:
        print("No pretrained model found")
        return None
