import numpy as np 
from pathlib import Path
import yaml     
from transformers import RobertaTokenizer, RobertaModel
from typing import Union, List 
import torch 
import time 

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
        
    def encode(
        self, code_batch: Union[list, str], query_batch: Union[list, str], language: str, batch_size : int = 32
    ) -> dict:        
        pass


    def load_tokenizer(self): 
        '''
        '''
        pass 

    def make_inference_minibatch(
        self,
        string_batch: Union[list, str],
        max_length_tokenizer: int,
        language: str,
        embedding_type: str,
    ) -> list:
        """
        Takes in a either a single string of a code or a query or a small batch, and returns an embedding for each input.
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
        start = time.time()
        model = self.model

        code_token_ids = self.tokenize(
            string_batch, max_length=max_length_tokenizer, mode="<encoder-only>"
        )
        with torch.no_grad():
            code_source_ids = torch.tensor(code_token_ids).to(self.device)
            inference_embeddings = (
                self.change_embedding_dtype(model.forward(code_inputs=code_source_ids), embedding_type)
            )

        return inference_embeddings

    def make_inference_batch(
        self,
        string_batch: Union[list, str],
        max_length_tokenizer: int,
        language: str,
        embedding_type: str,
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
        start = time.time()
        model = self.model
        code_embeddings_list = []

        # Sort inputs by list
        split_code_batch = self.split_list_equal_chunks(
            string_batch, self.serving_batch_size
        )

        for minibatch in split_code_batch:
            code_token_ids = self.tokenize(
                minibatch, max_length=max_length_tokenizer, mode="<encoder-only>"
            )
            with torch.no_grad():
                code_source_ids = torch.tensor(code_token_ids).to(self.device)
                code_embeddings_list.append(
                    self.change_embedding_dtype(model.forward(code_inputs=code_source_ids), embedding_type)
                )
            del code_source_ids
            torch.cuda.empty_cache()

        inference_embeddings = [x for xs in code_embeddings_list for x in xs]
        return inference_embeddings
