import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple, Optional
import time

class GestureDetector:
    """
    MediaPipe를 사용한 실시간 손 제스처 인식 클래스
    """
    
    def __init__(self):
        # MediaPipe 설정
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 손 인식 모델 초기화
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2  # 양손 인식
        )
        
        # 웹캠 초기화
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        
        # 제스처 상태
        self.hand_landmarks = []
        self.gesture_data = {
            'volume': 0.5,      # 0.0 ~ 1.0
            'pan': 0.0,         # -1.0 ~ 1.0
            'reverb': 0.0,      # 0.0 ~ 1.0
            'delay': 0.0,       # 0.0 ~ 1.0
            'filter': 20000,    # Hz
            'howling': 0.0,     # 0.0 ~ 1.0
            'finger_count': 0   # 손가락 수
        }
        
        # 좌표 안정화를 위한 필터
        self.position_history = []
        self.history_size = 5
        
    def start_camera(self, camera_id: int = None) -> bool:
        """
        웹캠 시작 - 노트북 내장 카메라 우선 사용
        """
        try:
            # 사용 가능한 카메라 찾기
            if camera_id is None:
                camera_id = self._find_builtin_camera()
            
            self.cap = cv2.VideoCapture(camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            if not self.cap.isOpened():
                print(f"카메라 ID {camera_id}를 열 수 없습니다. 다른 카메라를 시도합니다.")
                return self._try_other_cameras()
                
            print(f"웹캠 시작 완료 (카메라 ID: {camera_id})")
            return True
            
        except Exception as e:
            print(f"웹캠 시작 실패: {e}")
            return False
    
    def _find_builtin_camera(self) -> int:
        """
        노트북 내장 카메라 찾기
        """
        # 일반적으로 내장 카메라는 ID 0
        for camera_id in [0, 1, 2]:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    print(f"사용 가능한 카메라 발견: ID {camera_id}")
                    return camera_id
        
        print("내장 카메라를 찾을 수 없습니다. 기본값 0을 사용합니다.")
        return 0
    
    def _try_other_cameras(self) -> bool:
        """
        다른 카메라 시도
        """
        for camera_id in [1, 2, 0]:
            try:
                self.cap = cv2.VideoCapture(camera_id)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                    print(f"카메라 ID {camera_id}로 시작 성공")
                    return True
            except:
                continue
        
        print("사용 가능한 카메라가 없습니다.")
        return False
    
    def stop_camera(self):
        """
        웹캠 중지
        """
        if self.cap:
            self.cap.release()
        
        # MediaPipe 정리 (에러 방지)
        try:
            if self.hands:
                self.hands.close()
        except:
            pass
            
        cv2.destroyAllWindows()
        print("웹캠 중지")
    
    def process_frame(self) -> Tuple[bool, np.ndarray, Dict]:
        """
        프레임을 처리하고 제스처 데이터 반환
        """
        if not self.cap or not self.cap.isOpened():
            return False, None, self.gesture_data
        
        ret, frame = self.cap.read()
        if not ret:
            return False, None, self.gesture_data
        
        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 손 인식
        results = self.hands.process(rgb_frame)
        
        # 제스처 분석
        self._analyze_gestures(results, frame)
        
        # 손 랜드마크 그리기
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return True, frame, self.gesture_data
    
    def _analyze_gestures(self, results, frame):
        """
        손 제스처를 분석하고 제어 데이터 업데이트
        """
        if not results.multi_hand_landmarks:
            return
        
        # 첫 번째 손만 분석 (주요 제어용)
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        
        # 랜드마크 좌표 추출
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * self.frame_width)
            y = int(landmark.y * self.frame_height)
            landmarks.append((x, y))
        
        # 손가락 수 계산
        finger_count = self._count_fingers(landmarks)
        
        # 집게 모양 감지
        is_pinch = self._detect_pinch_gesture(landmarks)
        
        # 손 위치 기반 제어
        if len(landmarks) >= 21:  # MediaPipe 손 랜드마크는 21개
            # 손바닥 중심점 (0번 랜드마크)
            palm_x, palm_y = landmarks[0]
            
            # Y 좌표로 볼륨 제어 (위로 올릴수록 볼륨 증가)
            normalized_y = 1.0 - (palm_y / self.frame_height)
            volume = np.clip(normalized_y, 0.0, 1.0)
            
            # X 좌표로 팬 제어 (왼쪽에서 오른쪽으로)
            normalized_x = (palm_x / self.frame_width) * 2 - 1  # -1 ~ 1
            pan = np.clip(normalized_x, -1.0, 1.0)
            
            # 손가락 수에 따른 이펙트 제어
            volume_control = 0.5  # 기본 볼륨
            howling = 0.0
            delay = 0.0
            filter_freq = 20000
            reverb = 0.0
            
            if is_pinch:
                # 집게 모양: 볼륨 조절
                volume_control = volume
            elif finger_count == 0:
                # 주먹: 음악 일시정지/재생 토글 (볼륨은 고정)
                pass
            elif finger_count == 2:
                # 2개: 딜레이 on/off
                delay = 1.0 if volume > 0.5 else 0.0
            elif finger_count == 3:
                # 3개: 코러스 on/off
                pass  # 코러스는 별도 처리
            
            # 제스처 데이터 업데이트
            self.gesture_data.update({
                'volume': volume_control,
                'pan': pan,
                'reverb': reverb,
                'delay': delay,
                'filter': filter_freq,
                'howling': howling,
                'finger_count': finger_count,
                'is_pinch': is_pinch
            })
    
    def _count_fingers(self, landmarks: List[Tuple[int, int]]) -> int:
        """
        손가락 수 계산
        """
        if len(landmarks) < 21:
            return 0
        
        # 각 손가락의 끝점과 관절점 인덱스
        finger_tips = [4, 8, 12, 16, 20]  # 엄지, 검지, 중지, 약지, 새끼
        finger_pips = [3, 6, 10, 14, 18]  # 각 손가락의 두 번째 관절
        
        finger_count = 0
        
        # 엄지 (특별 처리)
        if landmarks[4][0] < landmarks[3][0]:  # 엄지가 왼쪽으로 펴져있음
            finger_count += 1
        
        # 나머지 손가락들
        for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
            if landmarks[tip][1] < landmarks[pip][1]:  # 끝점이 관절보다 위에 있음
                finger_count += 1
        
        return finger_count
    
    def _detect_pinch_gesture(self, landmarks: List[Tuple[int, int]]) -> bool:
        """
        집게 모양(검지와 엄지가 만나는 모양) 감지
        """
        if len(landmarks) < 21:
            return False
        
        # 엄지 끝점 (4)와 검지 끝점 (8)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # 두 점 사이의 거리 계산
        distance = ((thumb_tip[0] - index_tip[0]) ** 2 + 
                   (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5
        
        # 거리가 일정 값 이하면 집게 모양으로 판단
        # 화면 크기에 따라 임계값 조정
        pinch_threshold = min(self.frame_width, self.frame_height) * 0.05
        
        return distance < pinch_threshold
    
    def get_gesture_data(self) -> Dict:
        """
        현재 제스처 데이터 반환
        """
        return self.gesture_data.copy()
    
    def get_hand_position(self) -> Optional[Tuple[int, int]]:
        """
        손의 현재 위치 반환 (손바닥 중심)
        """
        if not self.hand_landmarks or len(self.hand_landmarks) < 1:
            return None
        
        # 손바닥 중심점 반환
        return self.hand_landmarks[0]
    
    def is_hand_detected(self) -> bool:
        """
        손이 감지되었는지 확인
        """
        return len(self.hand_landmarks) > 0 