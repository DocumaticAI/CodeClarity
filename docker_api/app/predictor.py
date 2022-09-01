# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import glob
import json
import os
import time
from typing import Optional, Union

import numpy as np
import pandas as pd
import uvicorn
from fastapi import APIRouter, FastAPI, Response
from schema import GenerationResponse, ModelSchema

from codeclarity import CodeEmbedder

app = FastAPI()
controller = APIRouter()

preloaded_models = {}


@app.on_event("startup")
def startup_event():
    print("downloading wrapped class for finetuned embedding models-")
    base_model = os.environ["base_model"]
    preloaded_models["embedding_handler"] = CodeEmbedder(str(base_model))
    pass


@controller.get("/ping")
def ping():
    """SageMaker required method, ping heartbeat"""
    return Response(status_code=200)


@controller.post("/invocations", response_model=GenerationResponse, status_code=200)
async def transformation(payload: ModelSchema):
    """
    Make an inference on a set of code or query snippits to return a set of embeddings

    Parameters
    ----------
    payload - Pydantic.BaseClass:
        a validated json object containing source code and queries that embeddings need to be returned for

    Returns
    -------
    predictions : dict:
        a dictionary object with embeddings for all queries and code snippits that are parsed in the request
    """
    start = time.time()
    model = preloaded_models["embedding_handler"]
    if payload.language not in model.allowed_languages:
        response_msg = f"Language currently unsupported. Supported language types are {model.allowed_languages}, got {payload.language}"
        return Response(
            response_msg,
            status_code=400,
            media_type="plain/text",
        )

    if payload.task == "embedding":
        if payload.code_snippit is not None:
            code_embeddings = model.encode(
                code_samples=payload.code_snippit,
                language=payload.language,
                return_tensors="list",
            )
            print(
                f"""response logged- num_samples:{len(payload.code_snippit)},language_specified:{payload.language}, total_inference_time:{time.time() - start}, average_time_per_sample": {(time.time() - start) / len(payload.code_snippit)}"""
            )
        if payload.query is not None: 
            query_embeddings = model.encode(
                code_samples=payload.query,
                return_tensors="list",
            )
            print(
                f"""response logged- num_samples:{len(payload.query)},language_specified:{payload.language}, total_inference_time:{time.time() - start}, average_time_per_sample": {(time.time() - start) / len(payload.query)}"""
            )
        response_body = {
            "code_response" : code_embeddings,
            "query_response" : query_embeddings
        }
        return Response(content=json.dumps(response_body), media_type="application/json")

    else:
        return Response(
            "Task currently unsupported. Supported task types are embedding, got task {}".format(
                payload.task
            ),
            status_code=400,
            media_type="plain/text",
        )


app.include_router(controller)

if __name__ == "__main__":
    uvicorn.run(app=app, port=8080)
