from typing import Union, List, Optional 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
import time 

from base import AbstractTransformerEncoder

class InCoderEmbedding(AbstractTransformerEncoder):
    '''
    '''
    def __init__(self, base_model : str) -> None:
        super(InCoderEmbedding, self).__init__()
        assert base_model in list(self.model_args['Incoder']['allowed_base_models'].keys()), \
            f"Incoder embedding model must be in \
            {list(self.model_args['Incoder']['allowed_base_models'].keys())}, got {base_model}"

        self.base_model = base_model
        self.serving_batch_size = self.model_args['Incoder']['serving']['batch_size']
        self.allowed_languages = self.model_args['Incoder']['allowed_base_models'][self.base_model]

        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
         
    def make_inference_batch(self, string_batch: Union[list, str], max_length_tokenizer: int, return_tensors: Optional[str] = "torch"):
        '''
        '''
        model = self.model
        code_embeddings_list = []

        # Sort inputs by list
        split_code_batch = self.utility_handler.split_list_equal_chunks(
            string_batch, self.serving_batch_size
        )
        for minibatch in split_code_batch:
            code_tokens = self.tokenize(
                minibatch, max_length=max_length_tokenizer, mode="<encoder-only>"
            )
            with torch.no_grad():
                model_outputs = model(code_tokens.input_ids, output_hidden_states=True, return_dict=True)
                last_hidden_state = model_outputs.hidden_states[-1]
                
                weight_matrix = (
                    torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand(last_hidden_state.size())
                    .float().to(last_hidden_state.device)
                )
                # Get attn mask of shape [bs, seq_len, hid_dim]
                input_mask_expanded = (
                    code_tokens["attention_mask"]
                    .unsqueeze(-1)
                    .expand(last_hidden_state.size())
                    .float()
                )
                sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weight_matrix, dim=1)
                sum_mask = torch.sum(input_mask_expanded * weight_matrix, dim=1)
                embedding_batch = sum_embeddings / sum_mask

                code_embeddings_list.append(
                    self.utility_handler.change_embedding_dtype(embedding_batch, return_tensors)
                )
                del code_tokens
                torch.cuda.empty_cache()

        inference_embeddings = [x for xs in code_embeddings_list for x in xs]
        return inference_embeddings

    def make_inference_minibatch(self, string_batch: Union[list, str], max_length_tokenizer: int, return_tensors: Optional[str] = "torch"):
        model = self.model

        code_tokens = self.tokenize(
            string_batch, max_length=max_length_tokenizer, mode="<encoder-only>"
        )
        with torch.no_grad():
            model_outputs = model(code_tokens.input_ids, output_hidden_states=True, return_dict=True)
            last_hidden_state = model_outputs.hidden_states[-1]
            
            weight_matrix = (
                torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(last_hidden_state.size())
                .float().to(last_hidden_state.device)
            )
            # Get attn mask of shape [bs, seq_len, hid_dim]
            input_mask_expanded = (
                code_tokens["attention_mask"]
                .unsqueeze(-1)
                .expand(last_hidden_state.size())
                .float()
            )
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weight_matrix, dim=1)
            sum_mask = torch.sum(input_mask_expanded * weight_matrix, dim=1)
            embedding_batch = sum_embeddings / sum_mask

        return self.utility_handler.change_embedding_dtype(embedding_batch, return_tensors)

    def tokenize(self,
        inputs: Union[List[str], str],
        max_length=512,
        padding=True):
        '''
        '''
        if isinstance(inputs, str):
            inputs = [inputs]

        batch_tokens = self.tokenizer(inputs, padding=padding, truncation=True, max_length = max_length, return_tensors="pt")
        return batch_tokens

    def load_model(self):
        '''
        class for loading incoder model with which to generate embeddings. The 
        setting of cuda device is overridden from the base class to include a conditional 
        check on amount of VRAM, as the model will not be able to be loaded on smaller GPUS due
        to the number of parameters 
        '''
        start = time.time()
        model_device = torch.device('cuda') if torch.cuda.is_available() \
            and self.utility_handler.check_host_gpu_ram() > 16 else 'cpu'

        if model_device == torch.device('cuda') and self.base_model in ["facebook/incoder-6B"]:
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
        ))
        return model.eval().to(model_device)

    def load_tokenizer(self):
        '''
        '''
        tokenizer = AutoTokenizer.from_pretrained(self.model_args['Incoder']['base_tokenizer'])
        tokenizer.pad_token =  "<pad>"
        tokenizer.padding_side = "left"
        return tokenizer



class CodeT5Embedder(AbstractTransformerEncoder):
    """ 
    
    """

    def __init__(self, base_model : str):
        super(CodeT5Embedder, self).__init__()
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

            