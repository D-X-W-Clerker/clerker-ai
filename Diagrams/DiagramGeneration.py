## DiagramGeneration.py
## main 함수 실행 파일

import json
import subprocess
import os
from llama_cpp import Llama
from transformers import AutoTokenizer
from Diagrams.DiagramRecognition import load_model_and_tokenizer, load_json, recommend_diagram_type

## 지정된 다이어그램 종류에 맞는 mermaid 코드를 가져오는 함수
def get_mermaid_code(diagram_type, diagrams_data):
    if diagram_type in diagrams_data['diagrams']:
        example1 = diagrams_data['diagrams'][diagram_type].get('example1', '')
        example2 = diagrams_data['diagrams'][diagram_type].get('example2', '')
        return example1, example2
    else:
        print(f"다이어그램 종류 '{diagram_type}'이(가) 존재하지 않습니다.")
        return None, None

## Mermaid 코드를 이미지로 저장하는 함수
def save_mermaid_to_image(mermaid_code, chunk_num):
    mermaid_file = f"./Diagrams/mermaid/chunk_{chunk_num}.mmd"
    output_file = f"./Diagrams/mermaid/chunk_{chunk_num}.png"
    puppeteer_config = "./Diagrams/puppeteer-config.json"

    ## Mermaid 코드에 폰트와 색상 설정 코드 추가
    font_and_size_setting = """
    %%{init: {
        "theme": "base",
        "themeVariables": {
            "fontFamily": "Noto Sans CJK, sans-serif",
            "primaryColor": "#c6dafd",       
            "primaryTextColor": "#000000",   
            "primaryBorderColor": "#2a4e85", 
            "lineColor": "#dde4ff",          
            "edgeLabelBackground": "#c6dafd"
        }
    }}%%
    
    """
    
    with open(mermaid_file, 'w') as f:
        f.write(font_and_size_setting + mermaid_code.strip())

    try:
        subprocess.run(
            ["mmdc", "-i", mermaid_file, "-o", output_file, "-p", puppeteer_config, "-s", "2.0"],
            check=True
        )
        print(f"Chunk {chunk_num}의 Mermaid 코드가 {output_file}로 저장되었습니다.")
    except subprocess.CalledProcessError as e:
        print(f"Mermaid 이미지 변환에 실패했습니다: {e}")
        pass

## 청크 단위로 추천된 다이어그램을 사용하여 instruction 생성 및 처리 및 이미지 저장
def process_chunks(chunks, diagrams_data, model, tokenizer, prompt_template, generation_kwargs):
    for chunk in chunks:
        ## 다이어그램 추천 및 적합성 판단
        diagram_type, is_diagram_suitable = recommend_diagram_type([chunk], model, tokenizer, prompt_template, generation_kwargs)
        
        if not is_diagram_suitable:
            print(f"Chunk {chunk.get('chunk_num', 'unknown')}는 다이어그램이 적합하지 않아 생성하지 않습니다.")
            continue

        summarized_text = chunk.get("summary", "")
        if not summarized_text:
            print(f"Chunk {chunk.get('chunk_num', 'unknown')}: summarized_text 없음")
            continue

        mermaid_code1, mermaid_code2 = get_mermaid_code(diagram_type, diagrams_data)
        if not mermaid_code1 or not mermaid_code2:
            print(f"Mermaid 코드가 {diagram_type}에 대해 존재하지 않습니다.")
            continue

        instruction = generate_instruction(summarized_text, mermaid_code1, mermaid_code2)

        messages = create_messages(prompt_template, instruction)

        # prompt 생성
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        response = model(prompt, **generation_kwargs)
        generated_text = response['choices'][0]['text'][len(prompt):]

        mermaid_code_generated = extract_mermaid_code(generated_text)

        save_mermaid_to_image(mermaid_code_generated, chunk.get('chunk_num', 'unknown'))

## Instruction Generation
def generate_instruction(summarized_text, mermaid_code1, mermaid_code2):
    return f'''
            예시 1:
            {mermaid_code1}

            예시 2:
            {mermaid_code2}

            예시 코드를 참고하여 아래의 회의 내용에 대해 "짧게" 요약한 뒤, mermaid 코드를 새로 만들어줘. (한글사용)
            다른 주석은 달지말고, 오직 코드로만 답변해줘. 오류가 나지 않도록 코드를 정확하게 생서해줘.

            "{summarized_text}"
            '''

def create_messages(prompt_template, instruction):
    return [
        {"role": "system", "content": f"{prompt_template}"},
        {"role": "user", "content": f"{instruction}"}
    ]

def extract_mermaid_code(text):
    if '```' in text:
        start_index = text.find('```mermaid') + len('```mermaid')
        end_index = text.find('```', start_index)
        mermaid_code = text[start_index:end_index].strip()
    else:
        mermaid_code = text.strip()
    return mermaid_code

def diagram_gen(input_summary_json):
    model_id = 'models/models--MLP-KTLim--llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/snapshots/4e602ad115392e7298674e092d6f8b45138f1db7'
    model_path = 'models/models--MLP-KTLim--llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/snapshots/4e602ad115392e7298674e092d6f8b45138f1db7/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf'
    model, tokenizer = load_model_and_tokenizer(model_id, model_path)

    data = load_json(input_summary_json)
    diagrams_data = load_json('./Diagrams/mermaid_code.json')
    chunks = data.get("chunks", [])

    PROMPT = '''
    당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.
    You are a helpful AI assistant, you'll need to answer users' queries in a friendly and accurate manner.
    '''

    generation_kwargs = {
        "max_tokens": 1024,
        "stop": ["<|eot_id|>"],
        "top_p": 0.9,
        "temperature": 0.6,
        "echo": True,
    }

    process_chunks(chunks, diagrams_data, model, tokenizer, PROMPT, generation_kwargs)