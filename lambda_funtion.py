import os
import json
import boto3
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

def main():
    input_domain = 'IT'
    mp3_file_url = "./input_audio.mp3"

    input_audio_file = './STT/stt_audio/input_audio.mp3'
    output_txt_file = './STT/stt_text/stt_text.txt'

    input_stt_txt = output_txt_file
    output_chunk_dict = './Chunking/chunking_text.json'

    output_summary_json = './Keywords/summary.json'
    diagram_summary_json = './Diagrams/diagram_summary.json'
    output_report_md = './Keywords/report.md'
    input_summary_json = output_summary_json 

    s3_font_path = 'Keywords/NanumFontSetup_TTF_SQUARE_ROUND/NanumSquareRoundB.ttf'

    #download_from_s3(mp3_file_url, input_audio_file)

    keyword_boosting_domain = f'STT/stt_text/KeywordBoosting/{input_domain}_KeywordBoosting.json'
    keyword_boosting_agenda = 'STT/stt_text/KeywordBoosting/Agenda_middle.json'

    download_from_s3(keyword_boosting_domain, keyword_boosting_domain)
    download_from_s3(keyword_boosting_agenda, keyword_boosting_agenda)
    print("boosting done")

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
        print(f"{s3_model_folder} done")

    download_from_s3(s3_font_path, s3_font_path)
    print("font done")

    make_stt_txt(
        input_domain,
        input_audio_file,
        output_txt_file,
    )
    print(f"STT 파일 생성 완료 : {output_txt_file}")

    make_chunk(output_txt_file, output_chunk_dict)
    print(f"Chunk Dict 파일 생성 완료 : {output_chunk_dict}")

    generate_summary_jsons(
        output_chunk_dict,
        diagram_summary_json,
        output_summary_json,
        model_id=os.path.join(local_model_dir, 'models--MLP-KTLim--llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/snapshots/4e602ad115392e7298674e092d6f8b45138f1db7/'),
        model_path=os.path.join(local_model_dir, 'models--MLP-KTLim--llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/snapshots/4e602ad115392e7298674e092d6f8b45138f1db7/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf')
    )
    print(f"다이어그램 및 보고서용 JSON 파일 생성 완료 : {output_summary_json}")

    diagram_gen(diagram_summary_json)
    print("다이어그램 생성 완료")

    generate_report_from_json(output_summary_json, output_report_md)
    print(f"Report 파일 생성 완료 : {output_report_md}")

    upload_to_s3(output_txt_file, 'STT/stt_text/stt_text.txt')
    upload_to_s3(output_chunk_dict, 'Chunking/chunking_text.json')
    upload_to_s3(output_summary_json, 'Keywords/summary.json')
    upload_to_s3(output_report_md, 'Keywords/report.md')

    diagram_images_dir = 'Diagrams/mermaid'
    for root, dirs, files in os.walk(diagram_images_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_file_path = f'Diagrams/mermaid/{file}'
            upload_to_s3(local_file_path, s3_file_path)

#     response = {
#         "report": f"https://{bucket_name}.s3.amazonaws.com/Keywords/report.md",
#         "stt": f"https://{bucket_name}.s3.amazonaws.com/STT/stt_text/stt_text.txt",
#         "diagram_image": f"https://{bucket_name}.s3.amazonaws.com/Diagrams/mermaid.zip"
#     }

#     return {
#         "statusCode": 200,
#         "body": json.dumps(response)
#     }

if __name__=="__main__":
    main()
