#!/usr/bin/env python3
"""
🎼 손 제스처 기반 실시간 음악 제어 시스템

사용자의 손 움직임을 인식하여 MP3 음악의 볼륨과 특수효과를 실시간으로 조절합니다.
"""

import cv2
import numpy as np
import time
import sys
import os
from typing import Optional

# 프로젝트 모듈 임포트
from audio_controller import AudioController
from gesture_detector import GestureDetector

class GestureMusicController:
    """
    손 제스처 기반 음악 제어 시스템의 메인 클래스
    """
    
    def __init__(self):
        self.audio_controller = AudioController()
        self.gesture_detector = GestureDetector()
        self.is_running = False
        
        # UI 설정
        self.window_name = "Gesture Music Controller"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 600)
        
        self.is_gesture_paused = False
        self.last_pause_gesture = False
        self.last_howling_gesture = False
        self.last_delay_gesture = False
        self.last_chorus_gesture = False
        
    def start(self, audio_file_path: str):
        """
        시스템 시작
        """
        print("🎼 손 제스처 기반 음악 제어 시스템 시작")
        print("=" * 50)
        
        # 1. 오디오 파일 로드
        if not self._load_audio(audio_file_path):
            return False
        
        # 2. 웹캠 시작
        if not self._start_camera():
            return False
        
        # 3. 오디오 재생 시작
        if not self._start_audio():
            return False
        
        # 4. 메인 루프 시작
        self._main_loop()
        
        return True
    
    def _load_audio(self, audio_file_path: str) -> bool:
        """
        오디오 파일 로드
        """
        if not os.path.exists(audio_file_path):
            print(f"❌ 오디오 파일을 찾을 수 없습니다: {audio_file_path}")
            print("📁 지원 형식: MP3, WAV, FLAC")
            return False
        
        print(f"🎵 오디오 파일 로드 중: {audio_file_path}")
        if self.audio_controller.load_audio_file(audio_file_path):
            print("✅ 오디오 파일 로드 완료")
            return True
        else:
            print("❌ 오디오 파일 로드 실패")
            return False
    
    def _start_camera(self) -> bool:
        """
        웹캠 시작
        """
        print("📷 웹캠 시작 중...")
        if self.gesture_detector.start_camera():
            print("✅ 웹캠 시작 완료")
            return True
        else:
            print("❌ 웹캠 시작 실패")
            return False
    
    def _start_audio(self) -> bool:
        """
        오디오 재생 시작
        """
        print("🔊 오디오 재생 시작 중...")
        if self.audio_controller.start_playback():
            print("✅ 오디오 재생 시작 완료")
            return True
        else:
            print("❌ 오디오 재생 시작 실패")
            return False
    
    def _main_loop(self):
        """
        메인 루프 - 실시간 제스처 인식 및 음악 제어
        """
        print("🎮 제어 방법:")
        print("• ✊ 주먹: 음악 재생/정지 토글")
        print("• 🤏 집게 모양: 위/아래로 음량 조절")
        print("• ✌️ 2개 손가락: 딜레이(에코) ON/OFF")
        print("• 🤟 3개 손가락: 코러스 ON/OFF")
        print("• 🎚️ 손의 좌우 위치: 스테레오 팬 제어")
        print("• 'q' 키: 종료")
        print("\n🎵 음악 제어를 시작합니다!")
        
        self.is_running = True
        last_update_time = time.time()
        
        while self.is_running:
            # 프레임 처리
            success, frame, gesture_data = self.gesture_detector.process_frame()
            
            if not success:
                print("❌ 프레임 처리 실패")
                break
            
            # 제스처 데이터로 오디오 제어
            self._update_audio_controls(gesture_data)
            
            # UI 업데이트 (1초에 10번만)
            current_time = time.time()
            if current_time - last_update_time > 0.1:
                self._update_ui(frame, gesture_data)
                last_update_time = current_time
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 프로그램을 종료합니다.")
                break
            elif key == ord('p'):
                # 재생/일시정지 토글
                if self.audio_controller.is_playing:
                    self.audio_controller.stop_playback()
                    print("⏸️ 일시정지")
                else:
                    self.audio_controller.start_playback()
                    print("▶️ 재생")
        
        self.stop()
    
    def _update_audio_controls(self, gesture_data: dict):
        """
        제스처 데이터를 기반으로 오디오 제어 업데이트
        """
        finger_count = gesture_data['finger_count']
        # 0개 손가락: 음악 일시정지/재생 토글
        pause_gesture = (finger_count == 0)
        if pause_gesture and not self.last_pause_gesture:
            self.is_gesture_paused = not self.is_gesture_paused
            if self.is_gesture_paused:
                self.audio_controller.pause()
            else:
                self.audio_controller.resume()
        self.last_pause_gesture = pause_gesture

        # 2개 손가락: 하울링 on/off
        howling_gesture = (finger_count == 2)
        if howling_gesture and not self.last_howling_gesture:
            self.audio_controller.set_howling(1.0)
        elif not howling_gesture and self.last_howling_gesture:
            self.audio_controller.set_howling(0.0)
        self.last_howling_gesture = howling_gesture

        # 2개 손가락: 딜레이 on/off
        delay_gesture = (finger_count == 2)
        if delay_gesture and not self.last_delay_gesture:
            self.audio_controller.set_delay(1.0)
        elif not delay_gesture and self.last_delay_gesture:
            self.audio_controller.set_delay(0.0)
        self.last_delay_gesture = delay_gesture

        # 3개 손가락: 코러스 on/off
        chorus_gesture = (finger_count == 3)
        if chorus_gesture and not self.last_chorus_gesture:
            self.audio_controller.set_chorus(1.0)
        elif not chorus_gesture and self.last_chorus_gesture:
            self.audio_controller.set_chorus(0.0)
        self.last_chorus_gesture = chorus_gesture

        if self.is_gesture_paused:
            return  # 정지 상태에서는 이펙트/볼륨 등 제어하지 않음

        # 1개 손가락: 볼륨 조절 (집게 모양)
        if gesture_data.get('is_pinch', False):
            self.audio_controller.set_volume(gesture_data['volume'])
        
        # 팬 제어 활성화 (손의 좌우 위치)
        self.audio_controller.set_pan(gesture_data['pan'])
        
        # 나머지 효과는 모두 0으로
        self.audio_controller.set_reverb(0.0)
        self.audio_controller.set_filter_freq(20000)
    
    def _update_ui(self, frame: np.ndarray, gesture_data: dict):
        """
        깔끔한 텍스트 기반 UI 업데이트
        """
        # 전체 화면에 미묘한 그라데이션 오버레이
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
        
        # 상태 정보 (좌측 상단)
        status_x = 30
        status_y = 50
        
        # 재생 상태
        play_status = "⏸️ PAUSED" if self.is_gesture_paused else "▶️ PLAYING"
        play_color = (100, 100, 255) if self.is_gesture_paused else (100, 255, 100)
        cv2.putText(frame, play_status, (status_x, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, play_color, 2)
        
        # 볼륨
        volume_text = f"🔊 Volume: {gesture_data['volume']:.1f}"
        cv2.putText(frame, volume_text, (status_x, status_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
        
        # 딜레이 상태
        delay_status = "⏱️ Delay: ON" if gesture_data['finger_count'] == 2 else "⏱️ Delay: OFF"
        delay_color = (255, 100, 100) if gesture_data['finger_count'] == 2 else (150, 150, 150)
        cv2.putText(frame, delay_status, (status_x, status_y + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, delay_color, 2)
        
        # 코러스 상태
        chorus_status = "🎵 Chorus: ON" if gesture_data['finger_count'] == 3 else "🎵 Chorus: OFF"
        chorus_color = (100, 255, 255) if gesture_data['finger_count'] == 3 else (150, 150, 150)
        cv2.putText(frame, chorus_status, (status_x, status_y + 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, chorus_color, 2)
        
        # 팬 상태
        pan_text = f"🎚️ Pan: {gesture_data['pan']:+.1f}"
        cv2.putText(frame, pan_text, (status_x, status_y + 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2)
        
        # 제스처 가이드 (우측 하단)
        guide_x = frame.shape[1] - 250
        guide_y = frame.shape[0] - 150
        
        # 제목
        cv2.putText(frame, "🎮 Controls", (guide_x, guide_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 제스처 설명
        gestures = [
            ("✊", "Play/Pause"),
            ("🤏", "Volume"),
            ("✌️", "Delay"),
            ("🤟", "Chorus")
        ]
        
        for i, (icon, desc) in enumerate(gestures):
            y_pos = guide_y + 30 + i * 25
            cv2.putText(frame, f"{icon} {desc}", (guide_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 종료 안내 (우측 상단)
        cv2.putText(frame, "Q to Exit", (frame.shape[1] - 120, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # 프레임 표시
        cv2.imshow(self.window_name, frame)
    
    def stop(self):
        """
        시스템 정리
        """
        print("🛑 시스템 정리 중...")
        self.is_running = False
        
        # 오디오 중지
        self.audio_controller.stop_playback()
        
        # 웹캠 중지
        self.gesture_detector.stop_camera()
        
        # 윈도우 정리
        cv2.destroyAllWindows()
        
        print("✅ 시스템 정리 완료")

def main():
    """
    메인 함수
    """
    print("🎼 손 제스처 기반 실시간 음악 제어 시스템")
    print("=" * 50)
    
    # 오디오 파일 경로 확인
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # 기본 오디오 파일 경로 (사용자가 제공해야 함)
        audio_file = input("🎵 오디오 파일 경로를 입력하세요 (MP3, WAV, FLAC): ").strip()
        
        if not audio_file:
            print("❌ 오디오 파일 경로가 필요합니다.")
            print("사용법: python main.py <audio_file_path>")
            return
    
    # 컨트롤러 생성 및 시작
    controller = GestureMusicController()
    
    try:
        controller.start(audio_file)
    except KeyboardInterrupt:
        print("\n👋 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        controller.stop()

if __name__ == "__main__":
    main() 