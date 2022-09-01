from typing import List, Optional, Union

from pydantic import BaseModel


class ModelSchema(BaseModel):
    code_snippit: Optional[Union[list, str]]
    query: Optional[Union[list, str]]
    language: str
    task: Optional[str] = "embedding"
    response_max_len: Optional[int] = 64


class EmbeddingBatch(BaseModel):
    input_strings: Union[List[str], str]
    embeddings: List[List[float]]

class GenerationResponse(BaseModel):
    code_response: Union[EmbeddingBatch, None]
    query_response: Union[EmbeddingBatch, None]
