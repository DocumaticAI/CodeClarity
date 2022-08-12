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
    def load_model(self): 
        pass 

    @abstractmethod
    def encode(self):
        pass 
