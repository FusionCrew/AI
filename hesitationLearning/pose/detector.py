"""
Body Gesture Detection Module (Rule-based)
MediaPipe Pose 랜드마크를 기반으로 규칙(Rule)을 통해 망설임 관련 제스처를 감지합니다.

감지 대상:
1. 팔짱 끼기 (Crossed Arms)
2. 얼굴 만지기 (Touching Face)
3. 뒷짐 지기 (Hands Behind Back)
4. 손 멈춤 (Static Hands)
"""
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple
from collections import deque

class BodyGestureDetector:
    def __init__(self, history_len: int = 30):
        # MediaPipe Pose 초기화
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils  # 시각화용
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 손목 위치 히스토리 (최근 N 프레임)
        self.history_len = history_len
        self.left_wrist_history = deque(maxlen=history_len)
        self.right_wrist_history = deque(maxlen=history_len)
        
        # 임계값 설정
        self.STATIC_THRESHOLD = 0.05  # 손 움직임 분산 임계값
        self.FACE_TOUCH_THRESHOLD = 0.15 # 얼굴-손 거리 임계값
    
    def process_frame(self, image: np.ndarray) -> Dict[str, bool]:
        """
        이미지에서 포즈를 추출하고 제스처를 감지
        """
        results = self.pose.process(image)
        
        gestures = {
            "extract_success": False,
            "crossed_arms": False,
            "touching_face": False,
            "hands_behind": False,
            "static_hands": False,
        }
        
        # 시각화용 랜드마크 저장
        self.pose_landmarks = None
        
        if not results.pose_landmarks:
            return gestures
            
        gestures["extract_success"] = True
        self.pose_landmarks = results.pose_landmarks
        landmarks = results.pose_landmarks.landmark
        
        # 주요 랜드마크 좌표 가져오기
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # 1. 팔짱 끼기 감지 (Crossed Arms)
        # 왼쪽 손목이 오른쪽 몸통 쪽으로, 오른쪽 손목이 왼쪽 몸통 쪽으로 이동
        # X 좌표 교차 여부 확인 (화면 기준: 왼쪽이 x 작음)
        # 사람 기준 왼쪽 손목(화면 오른쪽) -> 화면 왼쪽으로 이동
        # 간단히: 손목이 반대쪽 팔꿈치 안쪽에 위치하는지 확인
        
        is_arms_crossed = False
        if (left_wrist.visibility > 0.5 and right_wrist.visibility > 0.5 and
            left_elbow.visibility > 0.5 and right_elbow.visibility > 0.5):
            
            # X 좌표 기준 교차 확인
            # 정상 차렷 자세: R_Wrist(작음) < R_Elbow ... L_Elbow < L_Wrist(큼)
            # 팔짱: R_Wrist가 L_Elbow 근처, L_Wrist가 R_Elbow 근처
            
            # 수평 거리 차이가 작고, Y 위치가 비슷하면 팔짱으로 간주
            wrist_dist = abs(left_wrist.x - right_wrist.x)
            elbow_dist = abs(left_elbow.x - right_elbow.x)
            
            # 손목이 팔꿈치 너비보다 훨씬 안쪽으로 들어와 겹쳐야 함
            if wrist_dist < elbow_dist * 0.5:
                # 손목 높이가 팔꿈치보다 약간 위에 있거나 비슷해야 함 (팔짱은 보통 가슴 높이)
                if left_wrist.y < left_elbow.y + 0.1 and right_wrist.y < right_elbow.y + 0.1:
                    is_arms_crossed = True
                    
        gestures["crossed_arms"] = is_arms_crossed
        
        # 2. 얼굴 만지기 (Touching Face)
        # 손목이 코(Nose) 근처에 있는지 확인
        is_touching_face = False
        if nose.visibility > 0.5:
            l_dist = self._dist(left_wrist, nose)
            r_dist = self._dist(right_wrist, nose)
            
            if (left_wrist.visibility > 0.5 and l_dist < self.FACE_TOUCH_THRESHOLD) or \
               (right_wrist.visibility > 0.5 and r_dist < self.FACE_TOUCH_THRESHOLD):
                is_touching_face = True
                
        gestures["touching_face"] = is_touching_face
        
        # 3. 뒷짐 지기 (Hands Behind Back)
        # 어깨와 팔꿈치는 보이는데, 손목이 안 보임 (또는 등 뒤로 가서 가려짐)
        # MediaPipe는 가려진 부위도 추정하지만 visibility가 낮게 나옴
        is_hands_behind = False
        if (left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5 and
            left_elbow.visibility > 0.5 and right_elbow.visibility > 0.5):
            
            # 손목 Visibility가 낮거나, Z 좌표가 어깨보다 뒤(값 큼)에 있음
            # MediaPipe Pose Z: 몸통 중심이 0, 카메라도 가까우면 음수, 멀면 양수
            # 등 뒤면 Z값이 커야 함
            if (left_wrist.z > left_shoulder.z + 0.1 or right_wrist.z > right_shoulder.z + 0.1):
                 is_hands_behind = True
                 
        gestures["hands_behind"] = is_hands_behind
        
        # 4. 손 멈춤 (Static Hands)
        # 히스토리 업데이트
        if left_wrist.visibility > 0.5:
            self.left_wrist_history.append((left_wrist.x, left_wrist.y))
        
        if right_wrist.visibility > 0.5:
            self.right_wrist_history.append((right_wrist.x, right_wrist.y))
            
        is_static = False
        if len(self.left_wrist_history) >= self.history_len // 2:
            l_var = self._compute_variance(self.left_wrist_history)
            r_var = self._compute_variance(self.right_wrist_history)
            
            # 두 손 모두 움직임이 적으면 Static
            if l_var < self.STATIC_THRESHOLD and r_var < self.STATIC_THRESHOLD:
                is_static = True
                
        gestures["static_hands"] = is_static
        
        return gestures

    def _dist(self, p1, p2) -> float:
        """두 점 사이의 유클리드 거리 (X, Y)"""
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

    def _compute_variance(self, history) -> float:
        """좌표들의 분산(움직임 정도) 계산"""
        if not history:
            return 0.0
        arr = np.array(history)
        # X, Y 각각의 분산 합
        var = np.var(arr, axis=0).sum()
        return float(var) * 1000  # 스케일 조정 (보기 편하게)

    def close(self):
        self.pose.close()
