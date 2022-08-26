from typing import Optional, Union, List

from pydantic import BaseModel


class ModelSchema(BaseModel):
    code_snippit: Optional[Union[list, str]]
    query: Optional[Union[list, str]]
    language: str
    task: Optional[str] = "embedding"
    response_max_len: Optional[int] = 64


class CodeBatch(BaseModel):
    code_strings: Union[List[str], str]
    code_embeddings: List[List[float]]


class QueryBatch(BaseModel):
    query_strings: Union[List[str], str]
    query_embeddings: List[List[float]]


class GenerationResponse(BaseModel):
    code_batch: CodeBatch
    query_batch: QueryBatch
