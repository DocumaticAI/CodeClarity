from email.mime import base
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from typing import Union, List, Optional
from models import codebert, codet5, incoder, unixcoder


class CodeEmbedder(object):
    """ """

    def __init__(self, base_model: str):
        super(CodeEmbedder, self).__init__()
        self.embedding_models = {
            "CodeBert": codebert.CodeBertEmbedder,
            "CodeT5": codet5.CodeT5Embedder,
            "Incoder": incoder.InCoderEmbedder,
            "UniXCoder": unixcoder.UniXCoderEmbedder,
        }
        self.base_model = base_model

        self.config_path = Path(__file__).parent / "models" / "config.yaml"
        self.model_args = yaml.safe_load(self.config_path.read_text())

        self.model_type = [
            config
            for config in list(self.model_args.keys())
            if base_model in list(self.model_args[config]["allowed_base_models"].keys())
        ][0]

        self.embedder = self.embedding_models[self.model_type](base_model=base_model)
        self.allowed_languages = self.embedder.allowed_languages
    def encode(
        self,
        code_samples: Union[str, List[str]],
        language: Optional[str] = None,
        batch_size: Optional[int] = 32,
        max_length_tokenizer_nl: Optional[int] = 256,
        return_tensors: Optional[str] = "tensor",
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

        Returns
        -------
        code_batch : dict
            input_string : List[str]
                all strings passed into the model
            embeddings : List[List[float]]
                a dense vector returned by the ML model
        """
        if language:
            assert (
                language in self.allowed_languages
            ), f"""the programming language you've passed was not one of the 
                languages in the training or fintuning set for this model; using 
                this model for language {language} is likely to lead to poor performance.
                """

        embeddings = self.embedder.make_inference_batch(
            string_batch=code_samples,
            max_length_tokenizer=max_length_tokenizer_nl,
            batch_size=batch_size,
            return_tensors=return_tensors,
        )

        return (
            {
                "input_strings": code_samples,
                "embeddings": embeddings,
            },
        )
