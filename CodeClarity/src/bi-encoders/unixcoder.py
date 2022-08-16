from email.mime import base
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

from base import AbstractTransformerEncoder

class UniXEncoderBase(nn.Module):
    def __init__(self, base_model : str):
        super(UniXEncoderBase, self).__init__()
        self.encoder = RobertaModel.from_pretrained(base_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model)

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs * code_inputs.ne(1)[:, :, None]).sum(1) / code_inputs.ne(
                1
            ).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs * nl_inputs.ne(1)[:, :, None]).sum(1) / nl_inputs.ne(
                1
            ).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)


class UniXCoderEmbedder(AbstractTransformerEncoder):
    """ 
    
    """

    def __init__(self, base_model : str):
        super(UniXCoderEmbedder, self).__init__()
        assert base_model in list(self.model_args['UniXCoder']['allowed_base_models'].keys()), \
            f"UniXCoder embedding model must be in \
            {list(self.model_args['UniXCoder']['allowed_base_models'].keys())}, got {base_model}"
        
        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.model_args['UniXCoder']['base_tokenizer']
        )
        self.config = RobertaConfig.from_pretrained(
            base_model
        )
        self.base_model = base_model
        self.serving_batch_size = self.model_args['UniXCoder']['serving']['batch_size']

        self.allowed_languages = self.model_args['UniXCoder']['allowed_base_models'][self.base_model]
        self.model = self.load_model()


    def tokenize(
        self,
        inputs: Union[List[str], str],
        mode="<encoder-only>",
        max_length=256,
        padding=True,
    ) -> list:
        """
        Convert string to token ids
        Parameters:
        * `inputs`- list of input strings.
        * `max_length`- The maximum total source sequence length after tokenization.
        * `padding`- whether to pad source sequence length to max_length.
        * `mode`- which mode the sequence will use. i.e. <encoder-only>, <decoder-only>, <encoder-decoder>
        """
        assert mode in ["<encoder-only>", "<decoder-only>", "<encoder-decoder>"]

        tokenizer = self.tokenizer

        if isinstance(inputs, str):
            inputs = [inputs]

        tokens_ids = []
        for x in inputs:
            tokens = tokenizer.tokenize(x)
            if mode == "<encoder-only>":
                tokens = tokens[: max_length - 4]
                tokens = (
                    [tokenizer.cls_token, mode, tokenizer.sep_token]
                    + tokens
                    + [tokenizer.sep_token]
                )
            elif mode == "<decoder-only>":
                tokens = tokens[-(max_length - 3) :]
                tokens = [tokenizer.cls_token, mode, tokenizer.sep_token] + tokens
            else:
                tokens = tokens[: max_length - 5]
                tokens = (
                    [tokenizer.cls_token, mode, tokenizer.sep_token]
                    + tokens
                    + [tokenizer.sep_token]
                )

            tokens_id = tokenizer.convert_tokens_to_ids(tokens)
            if padding:
                tokens_id = tokens_id + [self.config.pad_token_id] * (
                    max_length - len(tokens_id)
                )
            tokens_ids.append(tokens_id)
        return tokens_ids

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
            string_batch, max_length=max_length_tokenizer, mode="<encoder-only>"
        )
        
        with torch.no_grad():
            code_source_ids = torch.tensor(code_token_ids).to(self.device)
            inference_embeddings = (
                self.utility_handler.change_embedding_dtype(model.forward(code_inputs=code_source_ids), return_tensors)
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
                minibatch, max_length=max_length_tokenizer, mode="<encoder-only>"
            )
            with torch.no_grad():
                code_source_ids = torch.tensor(code_token_ids).to(self.device)
                code_embeddings_list.append(
                    self.utility_handler.change_embedding_dtype(model.forward(code_inputs=code_source_ids), return_tensors)
                )
            del code_source_ids
            torch.cuda.empty_cache()

        inference_embeddings = [x for xs in code_embeddings_list for x in xs]
        return inference_embeddings

    def load_model(self):
        """
        Abstract loader for loading models from disk into embedding models for each language
        Arguments
        ---------
        model_language (str):
            a programming language for which to do search. Currently, each language has its own model
        Returns
        -------
        model_to_load (BaseEncoder):
            an instance of a wrapped roberta model that has been finetuned on the codesearchnet corpus
        """
        start = time.time()
        model = UniXEncoderBase(base_model = self.base_model)
        model_to_load = model.module if hasattr(model, "module") else model

        print(
            "Search retrieval model for allowed_languages {} loaded correctly to device {} in {} seconds".format(
                self.allowed_languages, self.device, time.time() - start
            )
        )
        return model_to_load.to(self.device)

            