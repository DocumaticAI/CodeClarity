# CodeClarity- Code Embeddings Made Easy

# About CodeClarity

This repository contains [CodeClarity] a lightweight app for creating contextual embeddings of source code in a format that is optimized and designed with code search and understanding.
in mind. This repository is part of a larger application providing a free exploration into the documatic codesearch tools capabilities. 

## Installation

We recommend **Python 3.7** or higher, **[PyTorch 1.6.0](https://pytorch.org/get-started/locally/)** or higher and **[transformers v4.6.0](https://github.com/huggingface/transformers)** or higher. The code does **not** work with Python 2.
**Install with pip (not currently live, coming soon.)**

Install the *codclarity* with `pip`:

```
pip install -U sentence-transformers
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/DocumaticAI/CodeClarity) and install it directly from the source code:

````
pip install -e .
```` 

**PyTorch with CUDA**

If you want to use a GPU / CUDA, you must install PyTorch with the matching CUDA Version. Follow
[PyTorch - Get Started](https://pytorch.org/get-started/locally/) for further details how to install PyTorch.


## Getting Started
First download a pretrained code model.

````python
from encoder import CodeEmbedder
model = CodeEmbedder(base_model = "microsoft/unixcoder-base")
````

Then provide some code snippits to the model. These can be full functions that could 
be parsed by an Abstract Syntax Tree, or small snippits.

````python
code_snippits = ['def read_csvs(dir) : return [pd.read_csv(fp) for fp in os.listdir(dir)]',
    "def set_pytorch_device(): return torch.device('cuda') if torch.cuda.is_available() else 'cpu", 
    'read file from disk into pandas dataframe']
code_embeddings = model.encode(code_snippits)
````

And that's it! We now have a list of returned embeddings of default type numpy array.

````python
for code, embedding in zip(code_snippits, code_embeddings):
    print("Sentence:", code)
    print("Embedding:", embedding)
    print("")
````

# API Drop in 
This project additionally impliments a docker container that serves a python REST api with the package running in it to serve a given model. to automatically build the container with any of the supported models for code search by running the following 

```
git clone https://github.com/DocumaticAI/code-embeddings-api.git 
cd api && bash ./setup.sh
```

Equally, to run the API outside the docker container, just clone the repository, navigate to the API folder and run the API file directly 
```
git clone https://github.com/DocumaticAI/code-embeddings-api.git 
pip install -r requirements-dev.txt
cd api
python predictor.py
```

## Pre-Trained Models

We provide implimentations of a range of code embedding models that are currently the state of the art in various tasks, including code semantic search, code clustering, code program detection, synthesis and more.  Some models are general purpose models, while others produce embeddings for specific use cases. Pre-trained models can be loaded by just passing the model name: `CodeEmbedder('model_name')`.

## Currently supported models
- [CodeBERT (base model): A Pre-Trained Model for Programming and Natural Languages](https://huggingface.co/microsoft/codebert-base)
- [CodeBERT (python finetuned on codesearchnet corpus): A Pre-Trained Model for Programming and Natural Languages](https://huggingface.co/documatic/codebert-python-finetuned)
- [UniXcoder (base model): Unified Cross-Modal Pre-training for Code Representation](https://huggingface.co/microsoft/unixcoder-base)
- [UniXcoder (9 language varient): Unified Cross-Modal Pre-training for Code Representation](https://huggingface.co/microsoft/unixcoder-base-nine)
- [UniXcoder (unimodal varient): Unified Cross-Modal Pre-training for Code Representation](https://huggingface.co/microsoft/unixcoder-base-unimodal)
- [InCoder 1B parameter model: A Generative Model for Code Infilling and Synthesis](https://huggingface.co/facebook/incoder-1B)
- [InCoder 6B parameter model: A Generative Model for Code Infilling and Synthesis](https://huggingface.co/facebook/incoder-6B)


## Internals of docker API

CodeClarity is designed to be a simple, modular dockerized python application that can be used to optain dense vector representations of natrual language code queries, and source code jointly to empower semantic search of codebases. 

It is comprised of a lightweight, async fastapi application running on a guicorn webserver. On startup, any of the supported models will be injected into the container, converted to an optimized serving format, and run on a REST API. 

CodeClarity automatically handles checking for supported languages for code models, dynamic batching of both code and natrual language snippits and conversions of code models to torchscript all in an asyncronous manner! 


## <a name="help"></a>Publications

The following papers are implimented or used heavily in this repo and this project would not be possible without their work:

- [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)
- [UniXcoder: Unified Cross-Modal Pre-training for Code Representation](https://arxiv.org/abs/2203.03850) (EMNLP 2020)
- [InCoder: A Generative Model for Code Infilling and Synthesis](https://arxiv.org/abs/2204.05999)
- [A Conversational Paradigm for Program Synthesis](https://arxiv.org/pdf/2203.13474.pdf)
- [CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation](https://github.com/salesforce/CodeT5) (EMNLP 2021)


# About Documatic

[Documatic](https://www.documatic.com/) is the company that deliversa more efficient codebase in 5 minutes. While you focus on coding, Documatic handles, creates and deploys the documentation so it's always up to date.


## <a name="help"></a>Getting help

If you have any questions about, feedback for or a problem with Codeclarity:

- Read [documatic website](https://www.documatic.com/).
- Sign up for the [documatic Waitlist](https://documatic-website.vercel.app/waitlist).
- [File an issue or request a feature](https://github.com/DocumaticAI/Roadmap).