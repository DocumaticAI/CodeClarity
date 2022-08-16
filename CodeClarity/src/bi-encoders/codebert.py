import numpy as np 
from pathlib import Path
import yaml     
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from typing import Union, List, Optional
import torch 
import time 

from base import AbstractTransformerEncoder


class CodeBertEmbedder(AbstractTransformerEncoder):
    '''
    '''
    def __init__(self, base_model : str) -> None:
        super(CodeBertEmbedder, self).__init__()
        assert base_model in list(self.model_args['CodeBert']['allowed_base_models'].keys()), \
            f"UniXCoder embedding model must be in \
            {list(self.model_args['CodeBert']['allowed_base_models'].keys())}, got {base_model}"

        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.model_args['CodeBert']['base_tokenizer']
        )
        self.config = RobertaConfig.from_pretrained(
            base_model
        )

        self.base_model = base_model
        self.serving_batch_size = self.model_args['CodeBert']['serving']['batch_size']

        self.allowed_languages = self.model_args['CodeBert']['allowed_base_models'][self.base_model]
        self.model = self.load_model()

    def make_inference_minibatch(
        self,
        string_batch: Union[list, str],
        max_length_tokenizer: int,
        return_tensors : Optional[str] = "torch"
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
        model = self.model

        code_token_ids = self.tokenize(
            string_batch, max_length=max_length_tokenizer
        )
        with torch.no_grad():
            code_source_ids = torch.tensor(code_token_ids).to(self.device)
            inference_embeddings = (
                self.utility_handler.change_embedding_dtype(model.forward(code_source_ids)[1], return_type= return_tensors)
            )

        return inference_embeddings

    def make_inference_batch(
        self,
        string_batch: Union[list, str],
        max_length_tokenizer: int,
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
        model = self.model
        code_embeddings_list = []

        # Sort inputs by list
        split_code_batch = self.utility_handler.split_list_equal_chunks(
            string_batch, self.serving_batch_size
        )

        for minibatch in split_code_batch:
            code_token_ids = self.tokenize(
                minibatch, max_length=max_length_tokenizer
            )
            with torch.no_grad():
                code_source_ids = torch.tensor(code_token_ids).to(self.device)
                code_embeddings_list.append(
                    self.utility_handler.change_embedding_dtype(model.forward(code_source_ids)[1], return_type= return_tensors)
                )
            del code_source_ids
            torch.cuda.empty_cache()

        inference_embeddings = [x for xs in code_embeddings_list for x in xs]
        return inference_embeddings

    def tokenize(self,
        inputs: Union[List[str], str],
        max_length=512,
        padding=True):
        '''
        '''
        if isinstance(inputs, str):
            inputs = [inputs]

        tokens_ids = []
        for snippit in inputs:
                tokens = self.tokenizer.tokenize(snippit)
                tokens = tokens[: max_length - 4]
                tokens=[self.tokenizer.cls_token]+tokens+[self.tokenizer.sep_token]
                tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)

                if padding:
                    tokens_id = tokens_id + [self.config.pad_token_id] * (
                        max_length - len(tokens_id)
                    )
                tokens_ids.append(tokens_id)
        return tokens_ids

    def load_model(self):
        '''
        '''
        model = RobertaModel.from_pretrained(self.base_model)
        return model.to(self.device)
    
