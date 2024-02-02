#standard library
import json
import numpy as np
#fastapi imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
#embedding library
import openai
#gcp libraries
from google.cloud import storage
import vertexai
from vertexai.preview.generative_models import GenerativeModel

#model imports
from utils.utilities import generate, cosine_similarity
from utils.models import EmbedDocument, Question
from utils.prompt import CONST_PROMPT
from utils.config import Settings

settings = Settings()
storage_client = storage.Client()

# TODO(developer): Update and un-comment below lines
# project_id = "PROJECT_ID"
location = settings.gcp_location
vertexai.init(project=settings.gcp_project_id, location=settings.gcp_location)

gen_model = GenerativeModel("gemini-pro")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
async def root():
    return {'message: Hello World'}

@app.post('/embed')
async def embed_document(item: EmbedDocument):
    bucket = storage_client.bucket(settings.gcp_storage_bucket)
    blob = bucket.blob(item.blob)
    file_data = blob.download_as_bytes()
    generate(file_data, bucket, settings)
    return {'message': 'Embed successful'}


@app.post('/retrieve-index')
async def retrieve_index(item: Question):
    openai.api_key = settings.openai_api_key
    chat = gen_model.start_chat()
    bucket = storage_client.bucket(settings.gcp_embedding_bucket)
    blobs = storage_client.list_blobs(settings.gcp_embedding_bucket)
    names = [blob.name for blob in blobs]
    read_file = lambda x: json.loads(bucket.blob(x).download_as_string())
    json_files = list(map(read_file, names))

    embeddings = [np.array(obj['embedding']) for obj in json_files]
    texts = [obj['text'] for obj in json_files]

    query = item.query
    query_embeddings = openai.Embedding.create(input=[query], engine="text-embedding-ada-002")
    query_embeddings = np.array(query_embeddings['data'][0]['embedding'])
    
    
    arr = np.array([cosine_similarity(query_embeddings, embedding) for embedding in embeddings])
    arr_indexes = arr.argsort()[-3:][::-1]
    information = [texts[i] for i in arr_indexes]

    prompt_formatted = CONST_PROMPT.format(question = query, information = information)
    res = chat.send_message(prompt_formatted)
    return {'message': res.text}



    






