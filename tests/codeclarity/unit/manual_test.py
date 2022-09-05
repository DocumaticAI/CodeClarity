import code
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

sys.path.insert(
    0, str(Path(__file__).parents[4] / "CodeClarity" / "codeclarity" / "bi_encoders"),
)
from encoder import CodeEmbedder

if __name__ == "__main__":
    embedder = CodeEmbedder("microsoft/unixcoder-base")

    x = embedder.encode("foo")
