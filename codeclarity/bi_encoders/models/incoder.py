from typing import Union, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

from .base import AbstractTransformerEncoder


class InCoderEmbedder(AbstractTransformerEncoder):
    """ """

    def __init__(self, base_model: str) -> None:
        super(InCoderEmbedder, self).__init__()
        assert base_model in list(
            self.model_args["Incoder"]["allowed_base_models"].keys()
        ), f"Incoder embedding model must be in \
            {list(self.model_args['Incoder']['allowed_base_models'].keys())}, got {base_model}"

        self.base_model = base_model
        self.serving_batch_size = self.model_args["Incoder"]["serving"][
            "default_batch_size"
        ]
        self.allowed_languages = self.model_args["Incoder"]["allowed_base_models"][
            self.base_model
        ]

        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def make_inference_minibatch(
        self,
        string_batch: Union[list, str],
        max_length_tokenizer: int,
        return_tensors: Optional[str] = "torch",
    ):
        """
        Define the forward pass of the model for a batch of inputs to get a 1 to 1
        embedding for every string passed. Sequence embedding is a weighted average
        of token embeddings, as implimented in SGPT paper
        https://arxiv.org/abs/2202.08904

        Parameters
        ----------
        string batch : Union[List[str], str]
             a list of string inputs of code or NL to be tokenized

        max_length_tokenizer : int
            a maximum tokenization length to be passed tokenize class method. For XGLM,
            can be set to at most 2048.

        return_tensors : Optional[str]
            the return format for the embeddings. Can be any of numpy, torch, tensorflow,
            tensor, list.
        """
        model = self.model

        code_tokens = self.tokenize(
            string_batch, max_length=max_length_tokenizer, mode="<encoder-only>"
        )
        with torch.no_grad():
            model_outputs = model(
                code_tokens.input_ids, output_hidden_states=True, return_dict=True
            )
            last_hidden_state = model_outputs.hidden_states[-1]

            weight_matrix = (
                torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(last_hidden_state.size())
                .float()
                .to(last_hidden_state.device)
            )
            # Get attn mask of shape [bs, seq_len, hid_dim]
            input_mask_expanded = (
                code_tokens["attention_mask"]
                .unsqueeze(-1)
                .expand(last_hidden_state.size())
                .float()
            )
            sum_embeddings = torch.sum(
                last_hidden_state * input_mask_expanded * weight_matrix, dim=1
            )
            sum_mask = torch.sum(input_mask_expanded * weight_matrix, dim=1)
            embedding_batch = sum_embeddings / sum_mask

        return self.utility_handler.change_embedding_dtype(
            embedding_batch, return_tensors
        )

    def tokenize(
        self, inputs: Union[List[str], str], max_length: int = 512, padding: bool = True
    ):
        """
        Take in a number of string inputs and tokenize them using the XGLM model tokenizer

        Arguments
        ---------
        inputs : Union[List[str], str]
             a list of string inputs of code or NL to be tokenized

        max_length : int
            the maximum length of a sequence to be tokenized as a torch tensor. For Incoder,
            maximum length can be set to 2048 as was the pretraining maximum.

        padding : bool
            bool on if to pad inputs if supplied in batch.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        batch_tokens = self.tokenizer(
            inputs,
            padding=padding,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return batch_tokens

    def load_model(self):
        """
        class for loading incoder model with which to generate embeddings. The
        setting of cuda device is overridden from the base class to include a conditional
        check on amount of VRAM, as the model will not be able to be loaded on smaller GPUS due
        to the number of parameters

        Returns
        -------
        model - XGLMModelForCausalLM:
            decoder only model arch that is used for code understanding from which
            we sample embeddings
        """
        start = time.time()
        model_device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            and self.utility_handler.check_host_gpu_ram() > 16
            else "cpu"
        )

        if model_device == torch.device("cuda") and self.base_model in [
            "facebook/incoder-6B"
        ]:
            kwargs = dict(
                gitrevision="float16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
        else:
            kwargs = dict(
                low_cpu_mem_usage=True,
            )
        model = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs).half()
        print(
            "Search retrieval model for allowed_languages {} loaded correctly to device {} in {} seconds".format(
                self.allowed_languages, self.device, time.time() - start
            )
        )
        return model.eval().to(model_device)

    def load_tokenizer(self):
        """ """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args["Incoder"]["base_tokenizer"]
        )
        tokenizer.pad_token = "<pad>"
        tokenizer.padding_side = "left"
        return tokenizer
