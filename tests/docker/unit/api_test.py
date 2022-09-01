import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, str(Path(__file__).parents[3] / "examples" / "docker_api" / "app"),
)
from predictor import app

@pytest.fixture
def payload():
    return {
  "code_snippit": [
    "string"
  ],
  "query": [
    "string"
  ],
  "language": "python",
  "task": "embedding",
  "response_max_len": 64
}

client = TestClient(app)

def test_read_main():
    response = client.get("/ping")
    assert response.status_code == 200

def test_api_unixcoder(payload):
    ##TODO 
    pass