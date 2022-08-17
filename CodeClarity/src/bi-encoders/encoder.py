from email.mime import base
import numpy as np
import pandas as pd 
import yaml 
from pathlib import Path

from typing import Union, List, Optional
from models import codebert, codet5, incoder, unixcoder
from models.base import AbstractTransformerEncoder

class CodeEmbedder(AbstractTransformerEncoder): 
    '''
    '''
    def __init__(self, base_model : str):
        super(CodeEmbedder, self).__init__()
        self.embedding_models = {
            "CodeBert" : codebert.CodeBertEmbedder,
            "CodeT5" : codet5.CodeT5Embedder,
            "Incoder" : incoder.InCoderEmbedder,
            "UniXCoder" : unixcoder.UniXCoderEmbedder
        }
        self.base_model = base_model
        self.model_type = [config for config in list(self.model_args.keys()) \
            if base_model in list(self.model_args[config]['allowed_base_models'].keys())]

        self.chosen_model = self.embedding_models[self.model_type](base_model = base_model)
        print(self.model_type)


    def load_model(self):
        return super().load_model()

    def make_inference_batch(self, string_batch: Union[list, str], max_length_tokenizer: int, return_tensors: Optional[str] = "torch"):
        return super().make_inference_batch(string_batch, max_length_tokenizer, return_tensors)
if __name__ == "__main__": 
    x = CodeEmbedder(base_model = "microsoft/unixcoder-base")
 