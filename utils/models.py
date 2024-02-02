from pydantic import BaseModel
from typing import Optional


class EmbedDocument(BaseModel):
    blob: str

class Question(BaseModel):
    query: str
