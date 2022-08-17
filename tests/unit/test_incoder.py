import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).parents[2]
        / "CodeClarity"
        / "src"
        /"bi-encoders"
    ),
)
from incoder import InCoderEmbedding

def test():
    pass

def test_unixcoder_embedding():
    embedder = InCoderEmbedding(base_model = "facebook/incoder-1B")
    assert embedder.encode(code_batch = "foo") is not None 

    return 1