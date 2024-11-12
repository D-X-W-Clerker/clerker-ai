import os
import json
import requests
import boto3
import zipfile
from STT.ClovaText import make_stt_txt
from Chunking.EmbeddingChunking import make_chunk
from Keywords.BllossomKeyword_to_md import generate_summary_jsons, generate_report_from_json
from Diagrams.DiagramGeneration import diagram_gen

ACCESS_KEY = "<>"
SECRET_KEY = "<>"
bucket_name = "<>"

s3 = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY
)

def download_from_s3(s3_path, local_path):
    s3.download_file(bucket_name, s3_path, local_path)

def upload_to_s3(local_path, s3_path):
    s3.upload_file(local_path, bucket_name, s3_path)

def download_folder_from_s3(s3_folder, local_dir):
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('/'):
                    continue

                relative_path = key[len(s3_folder):].lstrip('/')
                local_file_path = os.path.join(local_dir, relative_path)

                local_file_dir = os.path.dirname(local_file_path)
                os.makedirs(local_file_dir, exist_ok=True)

                s3.download_file(bucket_name, key, local_file_path)

def download_mp3(mp3_url, local_path):
    response = requests.get(mp3_url, stream=True)
    if response.status_code == 200:
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded MP3 file from {mp3_url} to {local_path}")
    else:
        raise Exception(f"Failed to download MP3 file from {mp3_url}. Status code: {response.status_code}")

def model_load():
    local_model_dir = 'models'
    os.makedirs(local_model_dir, exist_ok=True)

    model_folders = [
        'models--jhgan--ko-sroberta-sts/',
        'models--MLP-KTLim--llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/snapshots/4e602ad115392e7298674e092d6f8b45138f1db7/',
    ]
    for model_folder in model_folders:
        s3_model_folder = f'models/{model_folder}'
        local_model_folder = os.path.join(local_model_dir, model_folder)
        print(f"Downloading model folder from S3: {s3_model_folder}")
        download_folder_from_s3(s3_model_folder, local_model_folder)
        print(f"{s3_model_folder} download completed")

def input_fn(request_body, request_content_type='application/json'):
    return json.loads(request_body)

def zip_png_files(zip_path, source_dir):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, arcname=os.path.relpath(file_path, source_dir))
    print(f"Created ZIP file at {zip_path} with PNG files from {source_dir}")

def predict_fn(input_data, model):
    input_domain = input_data['domain']
    mp3_file_url = input_data['mp3FileUrl']

    model_load()
    input_audio_file = './STT/stt_audio/input_audio.mp3'
    output_txt_file = './STT/stt_text/stt_text.txt'
    output_chunk_dict = './Chunking/chunking_text.json'
    output_summary_json = './Keywords/summary.json'
    diagram_summary_json = './Diagrams/diagram_summary.json'
    output_report_md = './Keywords/report.md'
    s3_font_path = 'Keywords/NanumFontSetup_TTF_SQUARE_ROUND/NanumSquareRoundB.ttf'

    download_mp3(mp3_file_url, input_audio_file)

    download_from_s3(f'STT/stt_text/KeywordBoosting/{input_domain}_KeywordBoosting.json', 
                     f'STT/stt_text/KeywordBoosting/{input_domain}_KeywordBoosting.json')
    download_from_s3('STT/stt_text/KeywordBoosting/Agenda_middle.json', 
                     'STT/stt_text/KeywordBoosting/Agenda_middle.json')
    download_from_s3(s3_font_path, s3_font_path)

    make_stt_txt(input_domain, input_audio_file, output_txt_file)
    make_chunk(output_txt_file, output_chunk_dict)
    generate_summary_jsons(
        output_chunk_dict,
        diagram_summary_json,
        output_summary_json,
        model_id=os.path.join('models', 'models--MLP-KTLim--llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/snapshots/4e602ad115392e7298674e092d6f8b45138f1db7/'),
        model_path=os.path.join('models', 'models--MLP-KTLim--llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/snapshots/4e602ad115392e7298674e092d6f8b45138f1db7/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf')
    )
    diagram_gen(diagram_summary_json)
    generate_report_from_json(output_summary_json, output_report_md)

    upload_to_s3(output_txt_file, 'STT/stt_text/stt_text.txt')
    upload_to_s3(output_chunk_dict, 'Chunking/chunking_text.json')
    upload_to_s3(output_summary_json, 'Keywords/summary.json')
    upload_to_s3(output_report_md, 'Keywords/report.md')

    diagram_images_dir = 'Diagrams/mermaid'
    zip_path = './Diagrams/mermaid_images.zip'
    zip_png_files(zip_path, diagram_images_dir)

    upload_to_s3(zip_path, 'Diagrams/mermaid_images.zip')

    response = {
        "report": f"https://{bucket_name}.s3.amazonaws.com/Keywords/report.md",
        "stt": f"https://{bucket_name}.s3.amazonaws.com/STT/stt_text/stt_text.txt",
        "diagram_image": f"https://{bucket_name}.s3.amazonaws.com/Diagrams/mermaid_images.zip"
    }
    return response

def output_fn(prediction, content_type='application/json'):
    return json.dumps(prediction)
