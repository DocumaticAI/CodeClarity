import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

sys.path.insert(
    0, str(Path(__file__).parents[1] / "utils"),
)

from utils.processing import UtilityHandler


class AbstractTransformerEncoder(ABC):
    """
    Abstract class defining the encoding logic to be inherited for all derived classes.

    All Encapsulating classes must define a method 'load_model' to specify the model,
    and a forward pass method 'make_inference_minibatch'. Other than this, all other inference
    logic is handled dynamically.

    """

    allowed_languages: List[str]

    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.amp_device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.config_path = Path(__file__).parent / "config.yaml"
        self.model_args = yaml.safe_load(self.config_path.read_text())
        self.utility_handler = UtilityHandler

    @abstractmethod
    def tokenize(self):
        """
        Abstract method defined to require all inheriting bi-encoders to tokenize strings in a flexible manner
        """
        pass

    @abstractmethod
    def load_model(self):
        """
        Abstract method defined to require all inheriting bi-encoders to load a mode
        for inference in a flexible manner
        """
        pass

    @abstractmethod
    def make_inference_minibatch(
        self,
        string_batch: Union[list, str],
        max_length_tokenizer: int,
        return_tensors: Optional[str] = "torch",
    ):
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
        pass

    def make_inference_batch(
        self,
        string_batch: Union[list, str],
        max_length_tokenizer: int,
        batch_size: Optional[int] = 32,
        silence_progress_bar: Any = False,
        return_tensors: Optional[str] = "torch",
    ) -> Union[List[torch.tensor], List[np.array], List[List[int]]]:
        """
        Takes in a either a single string of a code or a query or a batch of any size, and returns an embedding for each input.
        Follows standard ML embedding workflow, tokenization, token tensor passed to model, embeddings
        converted to cpu and then turned to lists and returned, Most parameters are for logging.

        differs from method 'make_inference_minibatch' in that it encaptulates the forward pass logic,
        adding in memory management, loading bars ect..

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

        Returns
        -------
        inference_embeddings : Union[List[torch.tensor], List[np.array], List[List[int]]]]
            a data structure of a an embedding for every string in 'string_batch' passed into the method
        """

        batch_size = self.serving_batch_size if batch_size is not None else batch_size

        if isinstance(string_batch, str) or not hasattr(string_batch, "__len__"):
            string_batch = [string_batch]

        # Sort inputs by list
        code_embeddings_list = []
        length_sorted_idx = np.argsort(
            [-self.utility_handler.check_text_length(code) for code in string_batch]
        )
        sentences_sorted = [string_batch[idx] for idx in length_sorted_idx]

        # Sort inputs by list
        split_code_batch = self.utility_handler.split_list_equal_chunks(
            sentences_sorted, self.serving_batch_size
        )

        with tqdm(
            total=len(string_batch), file=sys.stdout, disable=silence_progress_bar
        ) as pbar:
                with torch.amp.autocast(device_type=self.amp_device, dtype=torch.bfloat16):
                    for batch in split_code_batch:
                        code_embeddings_list.extend(
                            self.make_inference_minibatch(
                                string_batch=batch,
                                max_length_tokenizer=max_length_tokenizer,
                                return_tensors=return_tensors,
                            ),
                        )
                        pbar.update(batch_size)

        inference_embeddings = [
            code_embeddings_list[idx] for idx in np.argsort(length_sorted_idx)
        ]

        return (
            inference_embeddings[0]
            if len(inference_embeddings) == 0
            else inference_embeddings
        )