import code
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
from unixcoder import UniXCoderEmbedder
from codebert import CodeBertEmbedder

def test():
    pass

def test_codebert_embedding():
    embedder = CodeBertEmbedder("microsoft/codebert-base")
    embed = embedder.encode(code_batch = "foo")
    assert embed is not None

"""def test_unixcoder_embedding():
    embedder = UniXCoderEmbedder(base_model = "microsoft/unixcoder-base")
    embedder.encode(code_batch = "foo")

    return None"""