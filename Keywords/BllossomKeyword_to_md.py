## BllossomKeyword_to_md.py
## main 함수 실행 파일

from llama_cpp import Llama
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
from transformers import AutoTokenizer
from collections import Counter
from wordcloud import WordCloud

import json
import re
import os

class AIAssistant:
    def __init__(self, model_id, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_gpu_layers=-1
        )
        self.prompt = (
            "당신은 유용한 AI 어시스턴트입니다. 사용자의 질의에 대해 친절하고 정확하게 답변해야 합니다.\n"
        )
        self.keyword_list = []

    def create_messages(self, instruction):
        return [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": instruction}
        ]

    ## Chunk Summarization
    ## Chunk별로 세 문장으로 요약하는 함수
    def generate_summary_from_chunk(self, chunk):
        original_text = chunk["original_text"]
        cleaned_text = clean_text(original_text)

        ## Summary Prompt
        prompt = (
            f"Original text: {cleaned_text}\n\n"
            "위 회의 텍스트를 아래의 형식에 맞게 요약해:\n"
            "제목: 해당 텍스트의 핵심 내용을 1~2 단어로 표현한 제목을 작성.\n"
            "핵심 키워드: 텍스트에서 중요한 키워드 2~3개를 작성.\n"
            "요약 3문장: 위 Original Text를 세 문장으로 요약.\n"
        )
        messages = self.create_messages(prompt)
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.model(inputs, max_tokens = 250, temperature=0, top_p=1)
        summary = outputs['choices'][0]['text'].strip()

        return summary
    

def generate_summary_jsons(input_chunk_dict, diagram_summary_json, report_summary_json, model_id, model_path):
    assistant = AIAssistant(model_id, model_path)

    with open(input_chunk_dict, 'r', encoding='utf-8') as f:
        data = json.load(f)

    summary_for_diagram_list = []
    report_summary_list = []
    keyword_for_wordcloud = ""

    for chunk in data['chunks']:
        keyword_for_wordcloud += clean_text(chunk['original_text'])
        summary = assistant.generate_summary_from_chunk(chunk)

        chunk_num = chunk['chunk_num']

        ## 제목 추출
        title_match = re.search(r"제목:\s*(.*)", summary)
        title = clean_text(title_match.group(1) if title_match else "")

        ## 키워드 추출
        keywords_match = re.search(r"핵심 키워드:\s*(.*)", summary)
        keywords = clean_text(keywords_match.group(1) if keywords_match else "")

        ## 요약 추출
        summary_match = re.search(r"요약 3문장:\s*(.*)", re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z .,?!:3]', '', summary.replace('1.', '').replace('2.', '').replace('3.', '')))
        if not summary_match:

            summary_match = re.search(r"(?<!발표 )요약:\s*(.*)", re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z .,?!:3]', '', summary.replace('1.', '').replace('2.', '').replace('3.', '')))
        full_summary = summary_match.group(1) if summary_match else ""
        summary_sentences = full_summary.split('.')
        summary_text = '. '.join(summary_sentences[:3]).replace('  ', ' ')

        chunk_dict_diagram = {
            "chunk_num": chunk_num,
            "summary": summary_text}
        summary_for_diagram_list.append(chunk_dict_diagram)

        chunk_dict_report = {
            "chunk_num": chunk_num,
            "title": title,
            "keywords": keywords,
            "summary": summary_text}
        report_summary_list.append(chunk_dict_report)

    with open(diagram_summary_json, 'w', encoding='utf-8') as json_file:
        json.dump({"chunks": summary_for_diagram_list}, json_file, ensure_ascii=False, indent=4)

    with open(report_summary_json, 'w', encoding='utf-8') as json_file:
        json.dump({"chunks": report_summary_list}, json_file, ensure_ascii=False, indent=4)

def generate_report_from_json(report_summary_json, output_report_md):
    with open(report_summary_json, 'r', encoding='utf-8') as json_file:
        report_data = json.load(json_file)

    markdown_result = ""
    chunk_titles = []  ## 소제목 리스트
    keyword_list = []  # 키워드 리스트

    ## md 생성
    for chunk in report_data['chunks']:
        chunk_num = chunk['chunk_num']
        title = chunk['title']
        summary_text = chunk['summary']
        chunk_titles.append(title)
        keyword_list.extend(chunk['keywords'].split(', '))

        image_path = f"Diagrams/mermaid/chunk_{chunk_num}.png"
        if os.path.exists(image_path):
            markdown_result += f"## {title.strip()}\n\n"
            markdown_result += f'![chunk_{chunk_num}]({image_path})\n\n'
            markdown_result += f"**요약**:\n\n{summary_text.strip()}.\n\n"
            markdown_result += "---\n\n"

    font_path = 'Keywords/NanumFontSetup_TTF_SQUARE_ROUND/NanumSquareRoundB.ttf'
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumSquareRound'

    keyword_counts_for_wordcloud = Counter(keyword_list)

    wordcloud = WordCloud(
        font_path=font_path,
        width=1000,
        height=600,
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(keyword_counts_for_wordcloud)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('키워드 빈도수 워드클라우드', size=15)
    plt.savefig('Keywords/keyword_wordcloud.png')
    plt.close()

    ## 가장 자주 등장하는 상위 5개의 키워드 추출
    common_keywords = Counter(keyword_list).most_common(5)
    keyword_text = ', '.join([kw[0] for kw in common_keywords])

    keywords, frequencies = zip(*common_keywords)
    plt.figure(figsize=(8, 5))
    plt.bar(keywords, frequencies, color='skyblue')
    plt.xlabel('키워드')
    plt.ylabel('빈도수')
    plt.title('상위 5개 키워드 빈도수')
    plt.savefig('Keywords/keyword_frequency.png')
    plt.close()

    ## 표지 작성
    cover_page = "# 전체 회의 요약 보고서\n\n"
    today_date = datetime.now().strftime("%Y-%m-%d")
    cover_page += f"#### 날짜 : {today_date}\n"
    cover_page += "## 상위 키워드\n\n"
    cover_page += keyword_text + "\n\n" 
    cover_page += "![상위 5개 키워드 빈도수](Keywords/keyword_frequency.png)\n\n"
    cover_page += "![키워드 워드클라우드](Keywords/keyword_wordcloud.png)\n\n"
    
    cover_page += "## 목차\n\n"
    for idx, title in enumerate(chunk_titles, start=1):
        cover_page += f"{idx}. {title}\n"

    with open(output_report_md, 'w', encoding='utf-8') as txt_file:
        txt_file.write(cover_page + "\n\n" + markdown_result)

def clean_text(text):
    return re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z .,?!]', '', text)