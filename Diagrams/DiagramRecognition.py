## DiagramRecognition.py
## 다이어그램 추천 함수

import json
from llama_cpp import Llama
from transformers import AutoTokenizer

def load_model_and_tokenizer(model_id, model_path, n_ctx=1024, n_gpu_layers=-1):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers
    )
    return model, tokenizer

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def recommend_diagram_type(chunks, model, tokenizer, prompt_template, generation_kwargs):
    for chunk in chunks:
        if not isinstance(chunk, dict):
            print(f"청크가 딕셔너리가 아님: {chunk}")
            continue

        summarized_text = chunk.get("summary", "")
        if not summarized_text:
            print(f"Chunk {chunk.get('chunk_num', 'unknown')}: summarized_text 없음")
            continue

        instruction = generate_recommendation_instruction(summarized_text)

        messages = create_messages(prompt_template, instruction)

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        response = model(prompt, **generation_kwargs)
        diagram_type, is_diagram_suitable = extract_diagram_type_and_suitability(response['choices'][0]['text'][len(prompt):])
        print(f"Chunk {chunk.get('chunk_num', 'unknown')} 추천 다이어그램: {diagram_type}, 다이어그램 적합성: {is_diagram_suitable}")

        return diagram_type, is_diagram_suitable

def generate_recommendation_instruction(summarized_text):
    return f'''
    "{summarized_text}"
    
    위의 요약한 회의 내용에 대해 아래의 다이어그램 중 회의 내용을 시각화하기 적합한 것으로 하나 추천해줘:
    Flowchart, Sequence Diagram, Class Diagram, Pie Chart, Quadrant Chart, Requirement Diagram, Gitgraph Diagram, Mindmaps, Timeline, XY Chart, Block Diagram.
    다이어그램으로 나타내기에 적합하지 않은 내용이라면, '다이어그램이 필요 없습니다.'라고 알려줘.
    '''

def create_messages(prompt_template, instruction):
    return [
        {"role": "system", "content": f"{prompt_template}"},
        {"role": "user", "content": f"{instruction}"}
    ]

def extract_diagram_type_and_suitability(response_text):
    if "다이어그램이 필요 없습니다" in response_text:
        return None, False
    
    for diagram_type in ["Flowchart", "Sequence Diagram", "Class Diagram", "Pie Chart", "Quadrant Chart", "Requirement Diagram", "Gitgraph Diagram", "Mindmaps", "Timeline", "XY Chart", "Block Diagram"]:
        # "Flowchart", "Sequence Diagram", "Class Diagram", "Pie Chart", "Quadrant Chart", "Requirement Diagram", "Gitgraph Diagram", "Mindmaps", "Timeline", "XY Chart", "Block Diagram"
        if diagram_type in response_text:
            return diagram_type, True

    return "추천 실패", False