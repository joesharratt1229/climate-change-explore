from pydantic import BaseSettings
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path = dotenv_path)

class Settings(BaseSettings):
    openai_api_key: str 
    gcp_project_id: str
    gcp_location: str
    gcp_storage_bucket: str
    gcp_embedding_bucket: str