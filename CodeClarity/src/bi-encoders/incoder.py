import numpy as np
import pandas as pd 
from typing import Union, List, Optional 
from transformers import AutoModelForCausalLM, AutoTokenizer

from base import AbstractTransformerEncoder

class InCoderEmbedding(AbstractTransformerEncoder):
    '''
    '''
    def __init__(self, base_model : str) -> None:
        super(InCoderEmbedding, self).__init__()
        assert base_model in list(self.model_args['Incoder']['allowed_base_models'].keys()), \
            f"UniXCoder embedding model must be in \
            {list(self.model_args['Incoder']['allowed_base_models'].keys())}, got {base_model}"

        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
         
        self.base_model = base_model
        self.serving_batch_size = self.model_args['Incoder']['serving']['batch_size']
        self.allowed_languages = self.model_args['Incoder']['allowed_base_models'][self.base_model]

    def load_model(self):
        '''
        class for loading incoder model with which to generate embeddings. The 
        setting of cuda device is overridden from the base class to include a conditional 
        check on amount of VRAM, as the model will not be able to be loaded on smaller GPUS due
        to the number of parameters 
        '''
        model = AutoModelForCausalLM.from_pretrained(self.base_model)


    def load_tokenizer(self):
        '''
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.model_args['Incoder']['base_tokenizer'])
        tokenizer.pad_token =  "<pad>"
        tokenizer.padding_side = "left"
        return tokenizer

    def make_inference_batch(self, string_batch: Union[list, str], max_length_tokenizer: int, return_tensors: Optional[str] = "torch"):
        pass

    def make_inference_minibatch(self, string_batch: Union[list, str], max_length_tokenizer: int, return_tensors: Optional[str] = "torch"):
        pass 
