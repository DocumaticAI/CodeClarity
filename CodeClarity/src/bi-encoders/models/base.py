import os
import time
from abc import abstractmethod
from pathlib import Path
from posixpath import split
from typing import Any, Dict, List, Optional, Union

import numpy as np 

import torch
import torch.nn as nn
import yaml
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
from abc import abstractmethod, ABC
import sys
from tqdm.autonotebook import trange

sys.path.insert(
    0,
    str(
        Path(__file__).parents[2]
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

    def make_inference_batch(
        self,
        string_batch: Union[list, str],
        max_length_tokenizer: int, 
        batch_size : Optional[int] = 32,
        show_tqdm_progress_bar : bool = None,
        return_tensors : Optional[str] = "torch"
    ) -> list:
        """
        Takes in a either a single string of a code or a query or a batch of any size, and returns an embedding for each input.
        Follows standard ML embedding workflow, tokenization, token tensor passed to model, embeddings
        converted to cpu and then turned to lists and returned, Most parameters are for logging.
        Parameters
        ----------
        string_batch - Union[list, str]:
            either a single example or a list of examples of a query or piece of source code to be embedded
        max_length_tokenizer - int:
            the max length for a snippit before it is cut short. 256 tokens for code, 128 for queries.
        language - str:
            logging parameter to display the programming language being inferred upon
        embedding_type - str:
            logging parameter to display the task for embedding, query or code.
        """
        batch_size = self.serving_batch_size if batch_size is not None else batch_size
        if isinstance(string_batch, str) or not hasattr(string_batch, '__len__'):
            string_batch = [string_batch]

        # Sort inputs by list
        code_embeddings_list = []
        length_sorted_idx = np.argsort([-self.utility_handler.check_text_length(code) for code in string_batch])
        sentences_sorted = [string_batch[idx] for idx in length_sorted_idx]

        split_code_batch = self.utility_handler.split_list_equal_chunks(
            string_batch, batch_size
        )

        for start_index in trange(0, len(string_batch), batch_size, desc="Batches", disable= not show_tqdm_progress_bar):
            sentence_batch = sentences_sorted[start_index:start_index+batch_size]
            
            code_embeddings_list.extend(
                    self.make_inference_minibatch(
                        string_batch= sentence_batch,
                        max_length_tokenizer= max_length_tokenizer,
                        return_tensors= return_tensors
                    ),
            )
            torch.cuda.empty_cache()

        inference_embeddings = [code_embeddings_list[idx] for idx in np.argsort(length_sorted_idx)]

        return inference_embeddings[0] if len(inference_embeddings) == 0 else inference_embeddings
