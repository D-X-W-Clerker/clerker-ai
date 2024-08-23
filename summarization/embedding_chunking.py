from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# embedding model name
## bert based ko embedding model
### 1. "jhgan/ko-sroberta-sts" (semantic textual similarity) 텍스트 의미적 유사성 -> 이게 괜찮아보임.
### 2. "jhgan/ko-sroberta-nli" (Natural Language Inference) 자연어 추론

model_name = "jhgan/ko-sroberta-sts" # 임베딩 모델 이름
number_of_chunks=10 # number of chunks: 총 몇 개의 청크로 나눌 것인지
max_length=512 # max_length: 각 청크의 최대 길이

def embedding(model_name):

    embeddings_model = HuggingFaceEmbeddings(

    model_name = model_name,
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True}
    
    )

    return embeddings_model

#### 청킹
## chunks를 리스트로 반환함.

# sementic chunker 사용

def semanticChunker(text , number_of_chunks ,max_length,text_splitter = SemanticChunker):
    if text_splitter == SemanticChunker:

        embeddings_model = embedding(model_name)
        text_splitter = text_splitter(embeddings=embeddings_model ,
                                        number_of_chunks=number_of_chunks, )
        
        docs = text_splitter.create_documents([text])

        chunks = []
        for doc in docs:

            ## SemanticChunker 사용 시, 청크 사이즈가 512 이상일 때 recursivecharacterTextSplitter 사용해서 max_length 이하로 다시 나눠줌
            if len(doc.page_content) > max_length:
                leaf_chunks = recursiveCharacterSplitter(doc.page_content,max_length)
                for leaf_chunk in leaf_chunks:
                    chunks.append(leaf_chunk)
            else:
                chunks.append(doc.page_content)

        return chunks

    else:
        docs = recursiveCharacterSplitter(text,max_length)
        
        chunks = []
        for doc in docs:
            chunks.append(doc.page_content)

        return chunks

# recursive charavter splitter 사용
def recursiveCharacterSplitter(text, max_length):
    text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size=max_length,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    )
    docs = text_splitter.create_documents([text])
    docs=[doc.page_content for doc in docs]
    return docs