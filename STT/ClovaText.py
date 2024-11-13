## ClovaText.py
## main 함수 실행 파일

import json
import os
from pydub import AudioSegment
from STT.ClovaSpeechClient import ClovaSpeechClient
from STT.AudioPreprocessing import AudioPreprocessing

def load_boosting_keywords(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data

def convert_to_wav(file_path):
    ## 파일이 wav 형식이 아닌 경우 wav 형식으로 변환
    file_ext = os.path.splitext(file_path)[-1].lower()
    
    if file_ext != ".wav":
        audio = AudioSegment.from_file(file_path)
        wav_file_path = os.path.splitext(file_path)[0] + ".wav"
        audio.export(wav_file_path, format="wav")
        print(f"파일이 {wav_file_path}로 변환되었습니다.")
        return wav_file_path
    return file_path

def make_stt_txt(input_domain,input_audio_file,output_txt_file):
    wav_audio_file = convert_to_wav(input_audio_file)
    
    ## Audio Preprocessing
    audio_preprocessor = AudioPreprocessing(wav_audio_file)
    audio_preprocessor.increase_volume().noise_reduction().normalize_volume().save_audio(wav_audio_file)

    ## Keyword Boosting
    domain_keyword = load_boosting_keywords(f'STT/stt_text/KeywordBoosting/{input_domain}_KeywordBoosting.json')
    agenda_keyword = load_boosting_keywords('STT/stt_text/KeywordBoosting/Agenda_middle.json')
    domain_keyword.extend(agenda_keyword)

    res = ClovaSpeechClient().req_upload(file=f'{wav_audio_file}', completion='sync', boostings = domain_keyword)
    result = res.json()

    ## Korean Stopwords
    stopwords = ["네", "네네", "네네네", "네네네네", "넵", "넷", "그", "아", "어", "저"]

    ## Formatting
    segments = result.get('segments', [])
    speaker_segments = []
    
    ## Speaker Numbering
    for segment in segments:
        speaker_label = segment['speaker']['label']
        text = segment['text']

        filtered_text = " ".join([word for word in text.split() if word not in stopwords])

        if len(filtered_text) > 3:
            speaker_segments.append(f"{speaker_label}: {filtered_text}")

    with open(output_txt_file, 'w', encoding='utf-8') as txt_file:
        txt_file.write("\n".join(speaker_segments))