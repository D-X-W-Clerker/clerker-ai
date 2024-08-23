import torch
import json
import sys
import os

#from kobart import get_kobart_tokenizer
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration
from langchain_experimental.text_splitter import SemanticChunker
from embedding_chunking import *

def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    # tokenizer = get_kobart_tokenizer()
    return model

def text_summarization(text):

    model = load_model()
    tokenizer = PreTrainedTokenizerFast.from_pretrained("hyunwoongko/kobart")
    #tokenizer = get_kobart_tokenizer()
    input_ids = tokenizer.encode(text)
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.unsqueeze(0)
    
    output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
    output = tokenizer.decode(output[0], skip_special_tokens=True)

    return output

def repetitive_summarization(chunks):
    output_dict = {"chunks":[]}
    for i in range(len(chunks)):
        #chunk = chunks[i]['page_content'].astype(str)
        chunk = chunks[i].replace('\n', '')
        summarized_text = text_summarization(chunk)
        output_dict["chunks"].append({"chunk_num":i,"original_text": chunk ,"summarized_text":summarized_text})

    return output_dict
                              
def save_to_json(file_path, content):

    with open(file_path, "w") as json_file:
        json.dump(content, json_file, ensure_ascii=False, indent=4)

    print(f"Summarized text saved to {file_path}")

if __name__ == "__main__":
    input_file_path = os.path.abspath('./stt_text.txt')

    with open(input_file_path, "r", encoding="utf-8") as file:
        input_text = file.read()

    # embedding_chunking
    chunks = semanticChunker(input_text, number_of_chunks, max_length, text_splitter=SemanticChunker)
    # repetitive_summarization
    summarized_text_dict = repetitive_summarization(chunks)

    output_file = "summarized_text.json"
    save_to_json(output_file, summarized_text_dict)