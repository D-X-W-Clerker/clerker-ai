# audio_processing.py
# 오디오 처리 관련 함수 모듈

import os
import torchaudio
from pydub import AudioSegment
from torchaudio.transforms import Resample

def convert_to_wav(audio_input):
    audio = AudioSegment.from_file(audio_input)
    output_file = audio_input.replace(".mp3", ".wav")
    audio.export(output_file, format="wav")
    return output_file

def load_audio(file_path):
    if file_path.endswith('.mp3'):
        file_path = convert_to_wav(file_path)
    
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

## 오디오 파일을 원하는 샘플레이트로 변환하는 함수
def resample_audio(waveform, orig_sample_rate, target_sample_rate):
    resampler = Resample(orig_freq=orig_sample_rate, new_freq=target_sample_rate)
    return resampler(waveform)
