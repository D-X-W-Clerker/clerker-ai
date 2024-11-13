## EmbeddingChunking.py
## main 함수 실행 파일

from tqdm import tqdm
import json
import sys
import os

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "models/models--jhgan--ko-sroberta-sts/snapshots/3efa8e54a06798b00bd1abb9c22b2dd530e22b24"
number_of_chunks=10  ## number of chunks
max_length=512  ## max_length

def embedding(model_name):

    embeddings_model = HuggingFaceEmbeddings(

    model_name = model_name,
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True}
    
    )
    return embeddings_model

## Sementic Chunker
def semanticChunker(text, number_of_chunks, max_length, text_splitter=SemanticChunker):
    if text_splitter == SemanticChunker:
        embeddings_model = embedding(model_name)
        text_splitter = text_splitter(embeddings=embeddings_model,
                                      number_of_chunks=number_of_chunks)
        
        docs = text_splitter.create_documents([text])

        chunks = []
        for doc in tqdm(docs, desc="Processing Chunking documents"):
            if len(doc.page_content) > max_length:
                leaf_chunks = recursiveCharacterSplitter(doc.page_content, max_length)
                for leaf_chunk in leaf_chunks:
                    chunks.append(leaf_chunk.page_content)
            else:
                chunks.append(doc.page_content)

        return chunks

    else:
        docs = recursiveCharacterSplitter(text, max_length)
        
        chunks = []
        for doc in tqdm(docs, desc="Processing RecursiveCharacterSplitter documents"):
            chunks.append(doc.page_content)

        return chunks

## Recursive Charavter Splitter
def recursiveCharacterSplitter(text, max_length):
    text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size=max_length,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )
    docs = text_splitter.create_documents([text])
    return docs

def sum_chunks(chunks):
    output_chunks = []
    pred_chunk = chunks[0].replace('\n', '')

    for i in range(1,len(chunks)):
        current_chunk = chunks[i].replace('\n', '')
        output_chunks.append(pred_chunk)
        if len(pred_chunk) + len(current_chunk) <= 512:
            current_chunk = pred_chunk + current_chunk
            output_chunks.pop()
        
        pred_chunk = current_chunk
        
    return output_chunks
                              
def save_to_json(file_path, content):
    
    with open(file_path, "w") as json_file:
        json.dump(content, json_file, ensure_ascii=False, indent=4)

    print(f"Summarized text saved to {file_path}")

def make_chunk(input_stt_txt, output_chunk_dict):

    with open(input_stt_txt, 'r') as file:
        input_text = file.readlines()
        
    pred_text = input_text[0].split(":")
    output_list = []
    for i in range(1,len(input_text)):
        current_text = input_text[i].split(":")
        output_list.append(":".join(pred_text))
        if pred_text[0].startswith(current_text[0]):
            current_text[1] = pred_text[1].replace('\n','') + current_text[1]
            output_list.pop()
        
        pred_text = current_text

    input = " ".join(output_list)

    ## embedding chunking
    number_of_chunks = len(input) // 500
    chunks = semanticChunker(text = input, number_of_chunks = number_of_chunks, max_length = 512, text_splitter=SemanticChunker)

    # sum chunks if chunk size is small
    chunks = sum_chunks(chunks)

    chunk_dict = {"chunks":[]}

    for i in range(len(chunks)):
        chunk = chunks[i]
        chunk_dict["chunks"].append({"chunk_num":i,"original_text": chunk })

    output_file = output_chunk_dict
    save_to_json(output_file, chunk_dict)