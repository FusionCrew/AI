"""
Hesitation Detection Model (LSTM)
망설임 감지 모델 - LSTM (Time-Series)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
from tqdm import tqdm

from .config import MODEL_PATH, MODEL_DIR, TRAIN_CONFIG

# GPU 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HesitationLSTM(nn.Module):
    """LSTM 기반 망설임 감지 모델"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=1):
        super(HesitationLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        
        # LSTM
        # out: (batch_size, seq_len, hidden_size)
        # hn: (num_layers, batch_size, hidden_size)
        out, (hn, cn) = self.lstm(x)
        
        # 마지막 시점의 은닉 상태 사용
        # out[:, -1, :]는 마지막 time step의 output
        out = out[:, -1, :]
        out = self.dropout(out)
        
        # Fully Connected
        out = self.fc(out)
        return out

class HesitationClassifier:
    """망설임 분류기 래퍼 (PyTorch 학습/추론 관리)"""
    
    def __init__(self, binary: bool = True):
        self.binary = binary
        self.model = None
        self.input_size = None
        self.history: Dict[str, List[float]] = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001,
            pos_weight: float = 1.0) -> Dict[str, float]:
        """
        모델 학습 (X는 (N, Time, Feature) 형태여야 함)
        """
        # 데이터 변환 (Numpy -> Tensor)
        X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE) # (N, 1)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 모델 초기화
        self.input_size = X_train.shape[2] # Feature dimension
        self.model = HesitationLSTM(input_size=self.input_size).to(DEVICE)
        
        # Class Weight 적용
        pos_weight_tensor = torch.tensor([pos_weight]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Validation 데이터 준비
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(DEVICE)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1).to(DEVICE)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"[INFO] Training on {DEVICE}...")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch) # (Batch, 1)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = correct / total
            self.history["loss"].append(avg_train_loss)
            self.history["accuracy"].append(train_acc)
            
            # Validation
            val_loss_str = ""
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        outputs = self.model(X_val_batch)
                        loss = criterion(outputs, y_val_batch)
                        val_loss += loss.item()
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        val_total += y_val_batch.size(0)
                        val_correct += (predicted == y_val_batch).sum().item()
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total
                self.history["val_loss"].append(avg_val_loss)
                self.history["val_accuracy"].append(val_acc)
                val_loss_str = f", Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
                
                # Early Stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Best model 저장 로직 추가 가능
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}{val_loss_str}")
                
        return {"accuracy": train_acc, "loss": avg_train_loss}

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """예측 (0 or 1)"""
        if self.model is None:
            raise ValueError("Model not trained")
            
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()
            
        return preds.cpu().numpy().flatten().astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """확률 예측"""
        if self.model is None:
            raise ValueError("Model not trained")
            
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs)
            
        return probs.cpu().numpy().flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """모델 평가"""
        y_pred = self.predict(X, threshold=threshold)
        
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, zero_division=0)
        }

    def save(self, path: Path = MODEL_PATH):
        """모델 저장"""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'history': self.history
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: Path = MODEL_PATH):
        """모델 로드"""
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
            
        checkpoint = torch.load(path, map_location=DEVICE)
        self.input_size = checkpoint['input_size']
        self.model = HesitationLSTM(input_size=self.input_size).to(DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from {path}")

    def get_training_history(self) -> Dict[str, List[float]]:
        return self.history

def load_pretrained_model(binary: bool = True) -> Optional[HesitationClassifier]:
    classifier = HesitationClassifier(binary=binary)
    try:
        classifier.load()
        return classifier
    except FileNotFoundError:
        return None
