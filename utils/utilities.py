import openai
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
import tiktoken
import fitz
import json
import openai
import numpy as np



def generate(file_data, bucket, settings):
    pdf_stream = BytesIO(file_data)
    whole_text = []
    pdf_document = fitz.open(stream = pdf_stream, filetype = 'pdf')
    for page in pdf_document:
        text = page.get_text()
        text = text.replace("\n", " ")
        whole_text.append(text)
        
    tokenizer = tiktoken.get_encoding("p50k_base")

    def tiktoken_len(text):
        tokens = tokenizer.encode(text, disallowed_special=())
        return len(tokens)


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""],)
    
    chunks = []

    for i, record in enumerate(whole_text):
        text_temp = text_splitter.split_text(record)
        for page_chunk in text_temp:
            chunks.append(page_chunk)

    for i, text_to_embed in enumerate(chunks):
        encode_texts_to_embeddings(text_to_embed, bucket, i, settings)



def encode_texts_to_embeddings(sentence: str, bucket, i, settings) -> None:
    openai.api_key = settings.openai_api_key
    embeddings = openai.Embedding.create(input=[sentence], engine="text-embedding-ada-002")
    embeddings = embeddings['data'][0]['embedding']
    temp_dict = {'id': i, 'embedding': embeddings, 'text': sentence}
    json_file = json.dumps(temp_dict, indent=4)
    blob_object = bucket.blob(f'lbg_{i}.json')
    blob_object.upload_from_string(json_file, content_type = 'application/json')

    

def cosine_similarity(arr1, arr2):
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    return (arr1 @ arr2)/(norm1 * norm2)
