import subprocess as sp 
import torch 
import numpy as np 
from typing import Union, List, Optional

class UtilityHandler(object):
    '''
    '''
    def __init__(self):
        self.single_device_memory = self.check_host_gpu_ram

    @staticmethod
    def check_host_gpu_ram():
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        return int(memory_free_info[0].strip("MiB ")) / 1000

    @staticmethod
    def split_list_equal_chunks(list_object, split_length):
        """Yield successive n-sized chunks from lst."""
        n = max(1, split_length)
        return (list_object[i:i+n] for i in range(0, len(list_object), n))

    @staticmethod
    def change_embedding_dtype(embedding : torch.Tensor, return_type : str): 
        '''
        Define the return dtype for the embedding
        '''
        allowed_return_types = ["np", "list", "tensor", "torch", "tf"]
        assert return_type in allowed_return_types, \
            f"Error, return type {return_type} provided. If overriding \
            return type, please specify an option from {allowed_return_types}"
        
        if return_type in ["tensor", "torch", "tf"]: 
            return [tensor for tensor in embedding.cpu().detach()]
        elif return_type == "np": 
            return [tensor for tensor in embedding.cpu().detach().numpy()]
        elif return_type == "list": 
            return [tensor for tensor in embedding.cpu().detach().tolist()]

    @staticmethod
    def check_text_length(text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])      #Sum of length of individual strings
