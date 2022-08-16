import os
import time
from abc import abstractmethod
from pathlib import Path
from posixpath import split
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import yaml
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from abc import abstractmethod, ABC
import sys

sys.path.insert(
    0,
    str(
        Path(__file__).parents[1]
        / "utils"
    ),
)

from processing import UtilityHandler

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
        self.utility_handler = UtilityHandler

    @abstractmethod
    def tokenize(self):
        pass 

    @abstractmethod
    def load_model(self): 
        pass 

    @abstractmethod
    def make_inference_minibatch(
        self,
        string_batch: Union[list, str],
        max_length_tokenizer: int,
        return_tensors : Optional[str] = "torch"
        ):
        pass

    @abstractmethod
    def make_inference_batch(
        self,
        string_batch: Union[list, str],
        max_length_tokenizer: int,
        return_tensors : Optional[str] = "torch"
        ):
        pass

    def encode(
        self, 
        code_batch: Union[list, str] = None, 
        query_batch: Union[list, str] = None, 
        language: Optional[str] = None, 
        batch_size : Optional[int] = 32, 
        max_length_tokenizer_nl : Optional[int] = 128, 
        max_length_tokenizer_pl : Optional[int] = 256,
        return_tensors : Optional[str] = "np"
    ) -> dict:
        """
        Wrapping function for making inference on batches of source code or queries to embed them.
        Takes in a single or batch example for code and queries along with a programming language to specify the
        language model to use, and returns a list of lists which corresponds to embeddings for each item in
        the batch.
        Parameters
        ----------
        code_batch - Union[list, str]:
            either a list or single example of a source code snippit to be embedded
        query_batch - Union[list, str]:
            either a list or single example of a query to be embedded to perform search
        language - str:
            a programming language that is required to specify the embedding model to use (each language that
            has been finetuned on has it's own model currently)
        """
        if language: 
            assert language in self.allowed_languages, \
                f"""the programming language you've passed was not one of the 
                languages in the training or fintuning set for this model; using 
                this model for language {language} is likely to lead to poor performance.
                """
        if code_batch:
            if (
                isinstance(code_batch, list)
                and len(code_batch) > batch_size
            ):
                code_embeddings = self.make_inference_batch(
                    string_batch=code_batch,
                    max_length_tokenizer=max_length_tokenizer_pl,
                    return_tensors = return_tensors
                )
            else:
                code_embeddings = self.make_inference_minibatch(
                    string_batch=code_batch,
                    max_length_tokenizer=max_length_tokenizer_pl,
                    return_tensors = return_tensors
                )
        else:
            code_embeddings = None

        if query_batch:
            if (
                isinstance(query_batch, list)
                and len(query_batch) > self.serving_batch_size
            ):
                query_embeddings = self.make_inference_batch(
                    string_batch=query_batch,
                    max_length_tokenizer=max_length_tokenizer_nl,
                    return_tensors = return_tensors
                )
            else:
                query_embeddings = self.make_inference_minibatch(
                    string_batch=query_batch,
                    max_length_tokenizer=max_length_tokenizer_nl,
                    return_tensors = return_tensors
                )
        else:
            query_embeddings = None

        return {
            "code_batch": {
                "code_strings": code_batch,
                "code_embeddings": code_embeddings,
            },
            "query_batch": {
                "query_strings": query_batch,
                "query_embeddings": query_embeddings,
            },
        }