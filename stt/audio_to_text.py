## audio_to_text.py
## 음성을 텍스트로 변환하는 함수 모듈

import os
import torch
from audio_processing import resample_audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from config import MODEL_NAME, PROCESSOR_NAME, HF_TOKEN, TARGET_SAMPLE_RATE

## 모델 및 프로세서를 정의하고 GPU에 태우는 함수
def setup_model():
    os.environ["HF_TOKEN"] = HF_TOKEN
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    processor = WhisperProcessor.from_pretrained(PROCESSOR_NAME, language="Korean", task="transcribe")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, processor, device

## 오디오 파일을 30초 단위로 분할하는 함수
def split_audio(waveform, original_sample_rate, chunk_duration_sec=30):
    chunk_size = chunk_duration_sec * original_sample_rate
    total_samples = waveform.size(1)
    return [waveform[:, i:i + chunk_size] for i in range(0, total_samples, chunk_size)]

## 30초 단위로 분할된 오디오 입력을 텍스트로 변환하는 함수
def generate_text_from_audio(model, processor, device, audio_input, orig_sample_rate):
    chunks = split_audio(audio_input, orig_sample_rate)
    full_transcription = []
    for chunk in chunks:
        resampled_chunk = resample_audio(chunk, orig_sample_rate, TARGET_SAMPLE_RATE).flatten()
        inputs = processor(resampled_chunk, return_tensors="pt", sampling_rate=TARGET_SAMPLE_RATE).input_features
        inputs = inputs.to(device)  # 입력 데이터를 GPU로 이동
        with torch.no_grad():
            predicted_ids = model.generate(inputs)
            transcription = processor.batch_decode(predicted_ids)
        full_transcription.extend(transcription)
    return ' '.join(full_transcription)
