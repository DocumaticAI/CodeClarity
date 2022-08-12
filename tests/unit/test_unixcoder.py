import code
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).parents[2]
        / "CodeClarity"
        / "src"
    ),
)
from unixcoder import UniXCoderEmbedder

def test():
    pass

def test_embedding():
    embedder = UniXCoderEmbedder(base_model = "microsoft/unixcoder-base")
    embedder.encode(code_batch = "foo")

    return None