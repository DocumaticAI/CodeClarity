import time
from typing import List, Optional, Union

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

from .base import AbstractTransformerEncoder


class UniXEncoderBase(nn.Module):
    def __init__(self, base_model: str):
        super(UniXEncoderBase, self).__init__()
        self.encoder = RobertaModel.from_pretrained(base_model)
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model)

    def forward(self, code_inputs : torch.tensor):
        '''
        forward pass of the UniXCoder model set 

        Arguments
        ---------
        code_inputs : torch.tensor
            either torch.tensor containing either a single or a list of tokenized
            string inputs for which to return an embedding
        
        Returns 
        -------
        A normalized torch tensor embedding for each input fed in the forward pass
        '''
        outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
        outputs = (outputs * code_inputs.ne(1)[:, :, None]).sum(1) / code_inputs.ne(
            1
        ).sum(-1)[:, None]
        return torch.nn.functional.normalize(outputs, p=2, dim=1)


class UniXCoderEmbedder(AbstractTransformerEncoder):
    """ """

    def __init__(self, base_model: str):
        super(UniXCoderEmbedder, self).__init__()
        assert base_model in list(
            self.model_args["UniXCoder"]["allowed_base_models"].keys()
        ), f"UniXCoder embedding model must be in \
            {list(self.model_args['UniXCoder']['allowed_base_models'].keys())}, got {base_model}"

        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.model_args["UniXCoder"]["base_tokenizer"]
        )
        self.config = RobertaConfig.from_pretrained(base_model)
        self.base_model = base_model
        self.serving_batch_size = self.model_args["UniXCoder"]["serving"][
            "default_batch_size"
        ]

        self.allowed_languages = self.model_args["UniXCoder"]["allowed_base_models"][
            self.base_model
        ]
        self.model = self.load_model()

    def tokenize(
        self,
        inputs: Union[List[str], str],
        mode : str="<encoder-only>",
        max_length : int =256,
        padding : bool=True,
    ) -> list:
        """
        Tokenize a series of 
        
        Parameters:
        inputs - Union[List[str], str]
            list of input strings.
        max_length - int 
            The maximum total source sequence length after tokenization
        padding - bool  
            whether to pad source sequence length to max_length.
        mode - Optional[str]
            which mode the sequence will use. i.e. <encoder-only>, <decoder-only>, <encoder-decoder>
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
        return_tensors: Optional[str] = "torch",
    ) -> list:
        """
        Define the forward pass of the model for a batch of inputs to get a 1 to 1 
        embedding for every string passed. The sentence embedding is the output of RobertaModel
        learned from pretraining and finetuning. 

        Parameters 
        ----------
        string batch : Union[List[str], str]
             a list of string inputs of code or NL to be tokenized

        max_length_tokenizer : int
            a maximum tokenization length to be passed tokenize class method. For RobertaModel,
            can be set to at most 512.
        
        return_tensors : Optional[str]
            the return format for the embeddings. Can be any of numpy, torch, tensorflow,
            tensor, list.
        """
        model = self.model

        code_token_ids = self.tokenize(
            string_batch, max_length=max_length_tokenizer, mode="<encoder-only>"
        )

        with torch.no_grad():
            code_source_ids = torch.tensor(code_token_ids).to(self.device)
            inference_embeddings = self.utility_handler.change_embedding_dtype(
                model.forward(code_inputs=code_source_ids), return_tensors
            )
        return inference_embeddings

    def load_model(self):
        """
        Loads RobertaModel from disk for embedding generation

        Returns
        -------
        model_to_load - BaseEncoder:
            an instance of a wrapped RobertaModel that has been finetuned on the codesearchnet corpus
        """
        start = time.time()
        model = UniXEncoderBase(base_model=self.base_model)
        model_to_load = model.module if hasattr(model, "module") else model

        print(
            "Search retrieval model for allowed_languages {} loaded correctly to device {} in {} seconds".format(
                self.allowed_languages, self.device, time.time() - start
            )
        )
        return model_to_load.to(self.device)
