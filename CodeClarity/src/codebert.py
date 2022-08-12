import numpy as np 
from pathlib import Path
import yaml     
from transformers import RobertaTokenizer, RobertaModel

from .base import AbstractTransformerEncoder

class CodeBert(AbstractTransformerEncoder):
    '''
    '''
    def __init__(self, base_model : str) -> None:
        super().__init__()
        assert base_model in list(self.model_args['CodeBert']['allowed_base_models'].keys()), \
            f"UniXCoder embedding model must be in \
            {list(self.model_args['CodeBert']['allowed_base_models'].keys())}, got {base_model}"

        self.tokenizer = RobertaTokenizer.from_pretrained(
            base_model
        )
        self.base_model = base_model
        self.serving_batch_size = self.model_args['CodeBert']['serving']['batch_size']

        self.allowed_languages = self.model_args['CodeBert']['allowed_base_models'][self.base_model]
        self.model = self.load_model()

    def load_model(self):
        '''
        '''
        model = RobertaModel.from_pretrained(self.base_model)
        return model.to(self.device)
        
    def encode(self, config : str) -> dict:
        '''
        #TODO
        '''
        pass

    def load_tokenizer(self): 
        '''
        '''
        pass 

    