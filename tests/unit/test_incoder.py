import code
import sys
from pathlib import Path
import pytest 
import torch 
import numpy as np
import time 
from typing import List 

sys.path.insert(
    0,
    str(
        Path(__file__).parents[2]
        / "CodeClarity"
        / "src"
        /"bi-encoders"
    ),
)
from encoder import CodeEmbedder

@pytest.fixture
def embedding_model():
    return CodeEmbedder(base_model = "facebook/incoder-1B")

def test_unixcoder_embedding(embedding_model):
    assert embedding_model.encode(code_samples = "foo") is not None 

def test_embedding_list_dtype(embedding_model):
    embeds = embedding_model.encode(code_samples = "foo", return_tensors = "list")['code_batch']['code_embeddings']
    assert isinstance(embeds[0], list)

def test_latency_batches(embedding_model):
    start = time.time()
    embeds =  embedding_model.encode(code_samples = ["foo" for x in range(128)], return_tensors = "numpy") is not None 
    assert time.time() - start < 16

def test_latency(embedding_model):
    time_batch = []
    for i in range(50):
        start = time.time()
        embeds =  embedding_model.encode(code_samples = ["foo" for x in range(1)], return_tensors = "numpy") is not None 
        time_batch.append(time.time() - start)
    assert np.mean(time_batch) < 0.05
