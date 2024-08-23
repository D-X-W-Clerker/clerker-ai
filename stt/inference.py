## main.py -> inference file
## 프로그램의 주 실행 파일

import os
import sys
from config import OUTPUT_TEXT_FILE, TARGET_SAMPLE_RATE
from audio_processing import load_audio, resample_audio
from audio_to_text import setup_model, generate_text_from_audio

def main(audio_file):
    audio_input = os.path.abspath(audio_file)
    model, processor, device = setup_model()

    waveform, orig_sample_rate = load_audio(audio_input)

    ## 텍스트 추출 및 저장
    transcription = generate_text_from_audio(model, processor, device, waveform, orig_sample_rate)
    
    with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as file:
        file.write(transcription)

    return transcription

if __name__ == "__main__":
    main(sys.argv[1])
