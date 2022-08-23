import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from posixpath import split
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import uvicorn
import yaml
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer


class AbstractTransformerEncoder(ABC):
    """
    class for the inheritance definitions for all of the encoders that will be usable as
    partof the public embeddings API.
    """

    allowed_languages: List[str]

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def encode(self):
        pass
