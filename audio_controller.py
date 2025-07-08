import sounddevice as sd
import numpy as np
import librosa
from scipy import signal
import threading
import queue
import time
from typing import Optional, Callable

class AudioController:
    """
    실시간 오디오 처리 및 제어를 담당하는 클래스
    sounddevice를 사용하여 지연 없는 실시간 음악 제어
    """
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_data = None
        self.current_position = 0
        self.is_playing = False
        self.volume = 1.0
        self.pan = 0.0  # -1.0 (왼쪽) ~ 1.0 (오른쪽)
        self.reverb_amount = 0.0
        self.delay_amount = 0.0
        self.filter_freq = 20000  # Hz
        self.howling_amount = 0.0  # 하울링 강도
        self.chorus_amount = 0.0  # 코러스 강도
        
        # 오디오 스트림 설정
        self.stream = None
        self.audio_queue = queue.Queue(maxsize=100)
        
        # 이펙트 버퍼
        self.reverb_buffer = np.zeros((sample_rate, channels))
        self.delay_buffer = np.zeros((sample_rate, channels))
        self.howling_buffer = np.zeros((sample_rate, channels))  # 하울링 버퍼
        self.chorus_buffer = np.zeros((sample_rate, channels))  # 코러스 버퍼
        self.reverb_pos = 0
        self.delay_pos = 0
        self.howling_pos = 0
        self.chorus_pos = 0
        
        # 스레드 안전을 위한 락
        self.lock = threading.Lock()
        
    def load_audio_file(self, file_path: str) -> bool:
        """
        MP3 파일을 로드하고 메모리에 저장
        """
        try:
            print(f"오디오 파일 로드 중: {file_path}")
            # librosa로 오디오 파일 로드
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=False)
            
            # 스테레오가 아닌 경우 스테레오로 변환
            if audio.ndim == 1:
                audio = np.tile(audio, (2, 1)).T
            elif audio.shape[0] == 2:  # (channels, samples) -> (samples, channels)
                audio = audio.T
                
            self.audio_data = audio
            self.current_position = 0
            print(f"오디오 로드 완료: {audio.shape[1]} 채널, {audio.shape[0]} 샘플")
            return True
            
        except Exception as e:
            print(f"오디오 파일 로드 실패: {e}")
            return False
    
    def start_playback(self):
        """
        실시간 오디오 재생 시작
        """
        if self.audio_data is None:
            print("재생할 오디오가 없습니다.")
            return False
            
        try:
            # 오디오 스트림 시작
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_playing = True
            print("실시간 오디오 재생 시작")
            return True
            
        except Exception as e:
            print(f"오디오 재생 시작 실패: {e}")
            return False
    
    def stop_playback(self):
        """
        오디오 재생 중지
        """
        self.is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        print("오디오 재생 중지")
    
    def pause(self):
        """
        오디오 재생 일시정지
        """
        with self.lock:
            self.is_playing = False
    
    def resume(self):
        """
        오디오 재생 재개
        """
        with self.lock:
            self.is_playing = True
    
    def _audio_callback(self, outdata, frames, time, status):
        """
        실시간 오디오 콜백 함수
        """
        if status:
            print(f"오디오 스트림 상태: {status}")
            
        if not self.is_playing or self.audio_data is None:
            outdata.fill(0)
            return
            
        with self.lock:
            # 현재 위치에서 프레임만큼 데이터 가져오기
            end_pos = min(self.current_position + frames, len(self.audio_data))
            if self.current_position >= len(self.audio_data):
                # 재생 완료
                outdata.fill(0)
                return
                
            # 오디오 데이터 추출
            audio_chunk = self.audio_data[self.current_position:end_pos].copy()
            
            # 프레임 수가 부족한 경우 0으로 패딩
            if len(audio_chunk) < frames:
                padding = np.zeros((frames - len(audio_chunk), self.channels))
                audio_chunk = np.vstack([audio_chunk, padding])
            
            # 볼륨 적용
            audio_chunk *= self.volume
            
            # 팬 적용
            if self.channels == 2:
                left_gain = np.sqrt(2) * np.cos((self.pan + 1) * np.pi / 4)
                right_gain = np.sqrt(2) * np.sin((self.pan + 1) * np.pi / 4)
                audio_chunk[:, 0] *= left_gain
                audio_chunk[:, 1] *= right_gain
            
            # 이펙트 적용
            audio_chunk = self._apply_effects(audio_chunk)
            
            # 출력 데이터 설정
            outdata[:] = audio_chunk.astype(np.float32)
            
            # 위치 업데이트
            self.current_position += frames
            
            # 재생 완료 시 처음부터 반복
            if self.current_position >= len(self.audio_data):
                self.current_position = 0
    
    def _apply_effects(self, audio_chunk):
        """
        실시간 이펙트 적용
        """
        # 하울링 적용
        if self.howling_amount > 0:
            audio_chunk = self._apply_howling(audio_chunk)
        
        # 코러스 적용
        if self.chorus_amount > 0:
            audio_chunk = self._apply_chorus(audio_chunk)
        
        # 리버브 적용
        if self.reverb_amount > 0:
            audio_chunk = self._apply_reverb(audio_chunk)
        
        # 딜레이 적용
        if self.delay_amount > 0:
            audio_chunk = self._apply_delay(audio_chunk)
        
        # 필터 적용
        if self.filter_freq < 20000:
            audio_chunk = self._apply_filter(audio_chunk)
        
        return audio_chunk
    
    def _apply_howling(self, audio_chunk):
        """
        하울링(피드백) 이펙트 적용
        """
        if self.howling_amount == 0:
            return audio_chunk
        
        chunk_len = len(audio_chunk)
        feedback = np.zeros_like(audio_chunk)
        for i in range(chunk_len):
            buffer_idx = (self.howling_pos + i) % len(self.howling_buffer)
            feedback[i] = self.howling_buffer[buffer_idx]
        # 효과 강도 증가 (0.8)
        howled_audio = audio_chunk + feedback * self.howling_amount * 0.8
        howled_audio = np.clip(howled_audio, -1.0, 1.0)
        for i in range(chunk_len):
            buffer_idx = (self.howling_pos + i) % len(self.howling_buffer)
            self.howling_buffer[buffer_idx] = howled_audio[i]
        self.howling_pos = (self.howling_pos + chunk_len) % len(self.howling_buffer)
        return howled_audio

    def _apply_chorus(self, audio_chunk):
        """
        코러스 이펙트 적용 (피치 시프트 기반)
        """
        if self.chorus_amount == 0:
            return audio_chunk
        
        # 간단한 코러스 구현 (피치 시프트 + 딜레이 조합)
        chorus_audio = np.copy(audio_chunk)
        
        # 약간의 피치 시프트 적용
        try:
            for channel in range(self.channels):
                # librosa를 사용한 피치 시프트 (약간만)
                shifted = librosa.effects.pitch_shift(
                    audio_chunk[:, channel], 
                    sr=self.sample_rate, 
                    n_steps=2  # 반음 정도만
                )
                
                # 길이 맞추기
                if len(shifted) > len(audio_chunk):
                    shifted = shifted[:len(audio_chunk)]
                elif len(shifted) < len(audio_chunk):
                    padding = np.zeros(len(audio_chunk) - len(shifted))
                    shifted = np.concatenate([shifted, padding])
                
                # 원본과 혼합
                chorus_audio[:, channel] += shifted * self.chorus_amount * 0.3
            
            return np.clip(chorus_audio, -1.0, 1.0)
            
        except Exception as e:
            print(f"코러스 적용 실패: {e}")
            return audio_chunk

    def _apply_reverb(self, audio_chunk):
        """
        간단한 리버브 효과 (강화됨)
        """
        reverb_samples = int(0.15 * self.sample_rate)  # 150ms 딜레이 (기존 100ms에서 증가)
        reverb_gain = self.reverb_amount * 0.8  # 0.3에서 0.8로 증가
        
        # 리버브 버퍼에 현재 오디오 추가
        for i in range(len(audio_chunk)):
            self.reverb_buffer[self.reverb_pos] = audio_chunk[i]
            self.reverb_pos = (self.reverb_pos + 1) % len(self.reverb_buffer)
            
            # 딜레이된 신호를 현재 신호에 추가
            delay_pos = (self.reverb_pos - reverb_samples) % len(self.reverb_buffer)
            audio_chunk[i] += self.reverb_buffer[delay_pos] * reverb_gain
        
        return audio_chunk
    
    def _apply_delay(self, audio_chunk):
        """
        딜레이 이펙트 적용
        """
        if self.delay_amount == 0:
            return audio_chunk
        delay_samples = int(self.sample_rate * 0.25)  # 250ms
        out = np.copy(audio_chunk)
        for i in range(len(audio_chunk)):
            idx = (self.delay_pos + i) % len(self.delay_buffer)
            delayed = self.delay_buffer[idx]
            # 효과 강도 대폭 감소 (0.3)
            out[i] += delayed * self.delay_amount * 0.3
            self.delay_buffer[idx] = out[i]
        self.delay_pos = (self.delay_pos + len(audio_chunk)) % len(self.delay_buffer)
        return np.clip(out, -1.0, 1.0)
    
    def _apply_filter(self, audio_chunk):
        """
        로우패스 필터 적용 (더 강한 효과)
        """
        # 버터워스 필터 설계 - 더 강한 필터링
        nyquist = self.sample_rate / 2
        normalized_freq = self.filter_freq / nyquist
        b, a = signal.butter(6, normalized_freq, btype='low')  # 4차에서 6차로 증가
        
        # 필터 적용
        for channel in range(self.channels):
            audio_chunk[:, channel] = signal.filtfilt(b, a, audio_chunk[:, channel])
        
        return audio_chunk
    
    def set_volume(self, volume: float):
        """
        볼륨 설정 (0.0 ~ 1.0)
        """
        with self.lock:
            self.volume = np.clip(volume, 0.0, 1.0)
    
    def set_pan(self, pan: float):
        """
        팬 설정 (-1.0 ~ 1.0)
        """
        with self.lock:
            self.pan = np.clip(pan, -1.0, 1.0)
    
    def set_reverb(self, amount: float):
        """
        리버브 강도 설정 (0.0 ~ 1.0)
        """
        with self.lock:
            self.reverb_amount = np.clip(amount, 0.0, 1.0)
    
    def set_delay(self, amount: float):
        """
        딜레이 강도 설정 (0.0 ~ 1.0)
        """
        with self.lock:
            self.delay_amount = np.clip(amount, 0.0, 1.0)
    
    def set_filter_freq(self, freq: float):
        """
        필터 주파수 설정 (Hz)
        """
        with self.lock:
            self.filter_freq = np.clip(freq, 20.0, 20000.0)
    
    def set_howling(self, amount: float):
        """
        하울링 강도 설정 (0.0 ~ 1.0)
        """
        with self.lock:
            self.howling_amount = np.clip(amount, 0.0, 1.0)
    
    def set_chorus(self, amount: float):
        """
        코러스 강도 설정 (0.0 ~ 1.0)
        """
        with self.lock:
            self.chorus_amount = np.clip(amount, 0.0, 1.0)
    
    def get_playback_info(self):
        """
        현재 재생 정보 반환
        """
        if self.audio_data is None:
            return None
            
        duration = len(self.audio_data) / self.sample_rate
        current_time = self.current_position / self.sample_rate
        
        return {
            'duration': duration,
            'current_time': current_time,
            'volume': self.volume,
            'pan': self.pan,
            'reverb': self.reverb_amount,
            'delay': self.delay_amount,
            'filter_freq': self.filter_freq,
            'howling': self.howling_amount,
            'chorus': self.chorus_amount,
            'is_playing': self.is_playing
        } 