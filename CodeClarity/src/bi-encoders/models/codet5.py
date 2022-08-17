import time 
from typing import Optional, List, Union 
from transformers import AutoTokenizer, T5ForConditionalGeneration

from .base import AbstractTransformerEncoder

class CodeT5Embedder(AbstractTransformerEncoder):
    '''
    '''
    def __init__(self, base_model : str): 
        super(CodeT5Embedder, self).__init__(self)
        pass
    