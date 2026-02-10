"""
Inference Module for Hesitation Detection
망설임 감지 추론 모듈 (Face LSTM + Body Gesture Rule-based)
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union, List
import base64

from .config import MODEL_PATH, MODEL_DIR, NUM_CLASSES, BINARY_THRESHOLD, MAX_SEQ_LEN
from .feature_extractor import FaceMeshExtractor, extract_features_from_video
from .model import HesitationClassifier, load_pretrained_model
from .pose import BodyGestureDetector


class HesitationDetector:
    """망설임 감지 추론 클래스 (LSTM + Body Gesture)"""
    
    def __init__(self, binary: bool = True):
        self.binary = binary
        self.classifier = None
        self.extractor = None
        self.body_detector = None
        self._load_model()
    
    def _load_model(self):
        """모델 및 추출기 초기화"""
        self.classifier = load_pretrained_model(binary=self.binary)
        if self.classifier is None:
            # 학습된 모델이 없으면 경고만 출력하고 진행
            print(f"[WARNING] No pretrained model found at {MODEL_PATH}")
        self.extractor = FaceMeshExtractor()
        self.body_detector = BodyGestureDetector()

    def _prepare_sequence(self, features: np.ndarray) -> np.ndarray:
        """
        특징 벡터를 LSTM 입력 형태 (1, MAX_SEQ_LEN, Feature)로 변환
        """
        # (Time, Feature) -> (1, MAX_SEQ_LEN, Feature)
        seq_len = len(features)
        feat_dim = features.shape[1]
        
        padded = np.zeros((1, MAX_SEQ_LEN, feat_dim), dtype=np.float32)
        
        if seq_len > MAX_SEQ_LEN:
            padded[0] = features[:MAX_SEQ_LEN]
        else:
            padded[0, :seq_len, :] = features
            
        return padded
    
    def detect_from_image(
        self, 
        image: np.ndarray,
        display: bool = False
    ) -> Dict[str, Union[int, float, str, Dict]]:
        """
        단일 이미지에서 망설임 감지 (Face + Body)
        Args:
            image: 입력 이미지 (BGR)
            display: True면 시각화된 이미지 반환 ("annotated_image" 키)
        """
        result = {
            "hesitation_level": -1,
            "confidence": 0.0,
            "label": "unknown",
            "body_gestures": {}
        }
        
        # 시각화용 복사본
        annotated_image = image.copy() if display else None

        # 1. Body Gesture Detection
        try:
            # Body Detector가 내부적으로 Pose 결과를 저장하고 있음
            body_gestures = self.body_detector.process_frame(image)
            result["body_gestures"] = body_gestures
            
            # 시각화 (Body Landmarks)
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
        
        # 2. Face Hesitation Detection
        # 랜드마크 추출
        landmarks = self.extractor.extract_landmarks(image)
        
        if landmarks is None:
            result["label"] = "no_face_detected"
            if display:
                result["annotated_image"] = annotated_image
            return result
            
        # 시각화 (Face Landmarks)
        # FaceLandmarkerResult는 face_landmarks 속성을 가짐 (List[List[NormalizedLandmark]])
        if display and self.extractor.results and self.extractor.results.face_landmarks:
            mp_drawing = self.extractor.mp_drawing
            mp_face_mesh = self.extractor.mp_face_mesh
            for face_landmarks in self.extractor.results.face_landmarks:
                # draw_landmarks는 NormalizedLandmarkList를 기대하지만, 
                # FaceLandmarker의 출력도 x,y,z 속성이 있어 호환될 수 있음.
                # 만약 호환되지 않는다면 Proto 변환이 필요할 수 있으나, 우선 속성명 수정.
                
                # mp_drawing.draw_landmarks를 쓰려면 landmark_list가 필요.
                # FaceLandmarker의 face_landmarks[i]는 List[NormalizedLandmark]임.
                # mp_drawing은 이를 보통 처리 가능.
                
                # 다만, draw_landmarks에 전달하기 위해 포맷을 맞춰야 할 수도 있음.
                # 여기서는 단순히 속성명만 변경하여 시도.
                
                # *주의*: mp_drawing.draw_landmarks는 landmark object가 .x .y .z .visibility 등을 가지길 기대함.
                # Tasks API의 NormalizedLandmark는 x,y,z를 가짐.
                
                # 리스트 형태이므로, 이를 그대로 넘겨주면 draw_landmarks가 처리 못할 수 있음 (Proto 기반이 아니라서).
                # 하지만 최신 mediapipe에서는 호환성을 위해 지원하는 경우가 많음.
                
                # 안전하게 그리기 위해 Proto 객체로 변환하거나, 반복문으로 그리는 방법이 있음.
                # 하지만 draw_landmarks가 가장 편하므로 일단 시도.
                
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=self._convert_to_proto(face_landmarks), # 변환 필요 가능성 있음
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(128, 256, 128), thickness=1, circle_radius=1)
                )
        
        if display:
            result["annotated_image"] = annotated_image
        
        # 특징 추출 (Dict -> List)
        features_dict = self.extractor.compute_hesitation_features(landmarks)
        feature_vector = np.array(list(features_dict.values())) # (Feature,)
        
        # LSTM 입력을 위해 시퀀스로 변환 (단일 이미지를 MAX_SEQ_LEN만큼 반복)
        features_seq = np.tile(feature_vector, (MAX_SEQ_LEN, 1)) # (150, 13)
        input_tensor = features_seq.reshape(1, MAX_SEQ_LEN, -1) # (1, 150, 13)
        
        # 예측
        if self.classifier:
            prediction = self.classifier.predict(input_tensor)[0]
            probabilities = self.classifier.predict_proba(input_tensor)[0]
            confidence = float(np.max(probabilities)) if isinstance(probabilities, np.ndarray) else float(probabilities)
            
            # 결과 업데이트
            fmt_res = self._format_result(prediction, confidence)
            result.update(fmt_res)
        else:
            # 모델이 없을 때 (아직 학습 전)
            result["hesitation_level"] = 0
            result["confidence"] = 0.0
            result["label"] = "model_not_loaded"
            
        return result

    def detect_from_video(
        self, 
        video_path: str
    ) -> Dict[str, Union[int, float, str, list, Dict]]:
        """
        비디오에서 망설임 감지
        """
        # 특징 추출 (Time, Feature)
        features = extract_features_from_video(video_path)
        
        if features is None:
            return {
                "hesitation_level": -1,
                "confidence": 0.0,
                "label": "video_error",
                "error": "Failed to extract features from video",
                "body_gestures": {}
            }
        
        # Body Gesture 분석 (마지막 부분 프레임 활용)
        body_gestures = {}
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # 너무 짧으면 처음부터, 길면 뒤쪽 10프레임 전 확인
            target_frame = max(0, total_frames - 10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if ret:
                body_gestures = self.body_detector.process_frame(frame)
            cap.release()
        except Exception as e:
            print(f"[ERROR] Body gesture video analysis failed: {e}")
        
        # LSTM 입력 준비 (Padding)
        input_tensor = self._prepare_sequence(features)
        
        # 예측
        prediction = 0
        confidence = 0.0
        
        result = {
            "hesitation_level": 0,
            "confidence": 0.0,
            "label": "unknown",
            "body_gestures": body_gestures
        }

        if self.classifier:
            prediction = self.classifier.predict(input_tensor)[0]
            probabilities = self.classifier.predict_proba(input_tensor)[0]
            confidence = float(np.max(probabilities))
            
            fmt_res = self._format_result(prediction, confidence)
            result.update(fmt_res)
        else:
            result["label"] = "model_not_loaded"
            
        return result

    def _format_result(self, prediction, confidence):
        """결과 포맷팅"""
        if self.binary:
            label = "hesitating" if prediction == 1 else "not_hesitating"
        else:
            labels = ["very_low", "low", "high", "very_high"]
            label = labels[prediction] if prediction < len(labels) else "unknown"
        
        return {
            "hesitation_level": int(prediction),
            "confidence": confidence,
            "label": label
        }
    
    def detect_from_base64(self, base64_image: str) -> Dict:
        """Base64 이미지 처리"""
        try:
            image_data = base64.b64decode(base64_image)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {"error": "Failed to decode image"}
            
            return self.detect_from_image(image)
        except Exception as e:
            return {"error": str(e)}

    def detect_from_file(self, file_path: str) -> Dict:
        """파일 이미지 처리"""
        image = cv2.imread(file_path)
        if image is None:
            return {"error": f"Failed to read image: {file_path}"}
        return self.detect_from_image(image)

    def _convert_to_proto(self, landmarks):
        """
        Tasks API의 NormalizedLandmark 리스트를 
        drawing_utils가 기대하는 NormalizedLandmarkList Proto로 변환
        """
        from mediapipe.framework.formats import landmark_pb2
        
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        
        for lm in landmarks:
            landmark = landmark_list.landmark.add()
            landmark.x = lm.x
            landmark.y = lm.y
            landmark.z = lm.z
            # visibility/presence는 있으면 추가
            if hasattr(lm, 'visibility'):
                landmark.visibility = lm.visibility
            if hasattr(lm, 'presence'):
                landmark.presence = lm.presence
                
        return landmark_list

    def close(self):
        """리소스 해제"""
        if self.extractor:
            self.extractor.close()
        if self.body_detector:
            self.body_detector.close()


# 싱글톤 인스턴스 (FastAPI에서 재사용)
_detector_instance: Optional[HesitationDetector] = None


def get_detector(binary: bool = False) -> HesitationDetector:
    """싱글톤 Detector 인스턴스 반환"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = HesitationDetector(binary=binary)
    return _detector_instance


def detect_hesitation(image: np.ndarray, binary: bool = False) -> Dict:
    """간편 함수"""
    detector = get_detector(binary=binary)
    return detector.detect_from_image(image)
