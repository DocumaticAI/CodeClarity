import os
import time
from abc import abstractmethod
from pathlib import Path
from posixpath import split
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import uvicorn
import yaml
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from abc import abstractmethod, ABC


class AbstractTransformerEncoder(ABC): 
    '''
    class for the inheritance definitions for all of the encoders that will be usable as 
    partof the public embeddings API. 
    ''' 
    allowed_languages : List[str]

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.config_path = Path(__file__).parent / "config.yaml"
        self.model_args = yaml.safe_load(self.config_path.read_text())

    @abstractmethod
    def tokenize(self):
        pass 

    @abstractmethod
    def load_model(self): 
        pass 

    @abstractmethod
    def encode(self):
        pass 

    @staticmethod
    def split_list_equal_chunks(list_object, split_length):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(list_object), split_length):
            yield list_object[i : i + split_length]

    @staticmethod
    def change_embedding_dtype(embedding : torch.Tensor, return_type : str): 
        '''
        Define the return dtype for the embedding
        '''
        allowed_return_types = ["np", "tensor", "list"]
        assert return_type in allowed_return_types, \
            f"Error, return type {return_type} provided. If overriding \
            return type, please specify an option from {allowed_return_types}"
        
        if return_type == "tensor": 
            return [tensor for tensor in embedding.cpu().detach()]
        elif return_type == "np": 
            return [tensor for tensor in embedding.cpu().detach().numpy()]
        elif return_type == "list": 
            return [tensor for tensor in embedding.cpu().detach().tolist()]
