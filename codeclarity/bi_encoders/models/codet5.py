import time
from typing import List, Optional, Union

from transformers import AutoTokenizer, T5ForConditionalGeneration

from .base import AbstractTransformerEncoder


class CodeT5Embedder(AbstractTransformerEncoder):
    """
    Coming Soon!
    """

    def __init__(self, base_model: str):
        super(CodeT5Embedder, self).__init__(self)
        pass
