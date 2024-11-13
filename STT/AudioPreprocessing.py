## AudioPreprocessing.py
## 오디오 파일 전처리 코드

import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter

class AudioPreprocessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.audio_data, self.sr = librosa.load(file_path, sr=None)

    def noise_reduction(self, noise_factor=0.005):
        ## Noise Reduction
        ## 오디오의 작은 random noise를 제거
        noise = np.random.randn(len(self.audio_data))
        augmented_data = self.audio_data + noise_factor * noise
        augmented_data = augmented_data.astype(type(self.audio_data[0]))
        self.audio_data = augmented_data
        return self

    def increase_volume(self, gain_factor=2):
        ## Volume Increase
        ## gain_factor만큼 볼륨을 증가
        self.audio_data = self.audio_data * gain_factor
        return self

    def normalize_volume(self):
        ## 음량을 1로 정규화
        max_volume = np.max(np.abs(self.audio_data))
        self.audio_data = self.audio_data / max_volume
        return self

    def high_pass_filter(self, cutoff=500, order=5):
        ## high-pass filter 적용
        ## cutoff 주파수 이상의 주파수만 통과
        def butter_highpass(cutoff, fs, order=5):
            nyquist = 0.5 * fs
            normal_cutoff = cutoff / nyquist
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a

        def highpass_filter(data, cutoff, fs, order=5):
            b, a = butter_highpass(cutoff, fs, order=order)
            y = lfilter(b, a, data)
            return y

        self.audio_data = highpass_filter(self.audio_data, cutoff, self.sr, order)
        return self

    def slow_down_audio(self, rate=0.9):
        ## 속도 느리게
        ## rate 비율로 속도를 줄임 (기본은 0.9배)
        self.audio_data = librosa.effects.time_stretch(self.audio_data, rate = rate)
        return self

    def save_audio(self, output_path):
        sf.write(output_path, self.audio_data, self.sr)