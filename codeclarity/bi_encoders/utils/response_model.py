from typing import Any, List, Union

import numpy as np
import torch
from pydantic import BaseModel


class EmbeddingResponseModel(BaseModel):
    embeddings: Union[List[List[float]], List[np.ndarray], List[torch.Tensor]]
    input_strings: Union[List[str], str]
    embedding_time: float
    batch_size: int
    model_used: str

    class Config:
        arbitrary_types_allowed = True
