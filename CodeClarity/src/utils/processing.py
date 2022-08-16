import subprocess as sp 
import torch 
import numpy as np 
 

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
        for i in range(0, len(list_object), split_length):
            yield list_object[i : i + split_length]

    @staticmethod
    def change_embedding_dtype(embedding : torch.Tensor, return_type : str): 
        '''
        Define the return dtype for the embedding
        '''
        allowed_return_types = ["np", "tensor", "list"]
        assert return_type in allowed_return_types, \
            f"Error, return type {return_type} provided. If overriding \
            return type, please specify an option from {allowed_return_types}"
        
        if return_type == "tensor": 
            return [tensor for tensor in embedding.cpu().detach()]
        elif return_type == "np": 
            return [tensor for tensor in embedding.cpu().detach().numpy()]
        elif return_type == "list": 
            return [tensor for tensor in embedding.cpu().detach().tolist()]

