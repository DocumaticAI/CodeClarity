from pydantic import BaseModel 
from typing import Union, List, Optional, Any
from torch import Tensor
import numpy as np
 
class EmbeddingResponseModel(BaseModel):
    embeddings : Union[List[List[float]], List[np.ndarray], List[Tensor]]
    input_strings : Union[List[str], str]
    embedding_time : float
    batch_size : int
    model_used : str

    class Config:
        arbitrary_types_allowed = True
