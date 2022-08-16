import numpy as np
import pandas as pd 
from typing import Union, List, Optional 
from transformers import AutoModelForCausalLM, AutoTokenizer
 
from base import AbstractTransformerEncoder

class InCoderEmbedding(AbstractTransformerEncoder):
    '''
    '''
    def __init__(self, base_model : str) -> None:
        self.encoder = RobertaModel.from_pretrained(base_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model)

    def load_model(self):
        pass

    def load_tokenizer(self):
        pass

    def make_inference_batch(self, string_batch: Union[list, str], max_length_tokenizer: int, return_tensors: Optional[str] = "torch"):
        pass

    def make_inference_minibatch(self, string_batch: Union[list, str], max_length_tokenizer: int, return_tensors: Optional[str] = "torch"):
        pass 
