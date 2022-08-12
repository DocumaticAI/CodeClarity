# CodeClarity- Code Embeddings Made Easy

# About Documatic

[Documatic](https://www.documatic.com/) is the company that deliversGet a more efficient codebase in 5 minutes. While you focus on coding, Documatic handles, creates and deploys the documentation so it's always up to date.

This repository contains [CodeClarity] a lightweight app for creating contextual embeddings of source code in a format that is optimized and designed with code search systems 
in mind. This repository is part of a larger application providing a free exploration into the documatic codesearch tools capabilities. 

# Internals 
#CodeClarity is designed to be a simple, modular dockerized python application that can be used to optain dense vector representations of natrual language code queries, and source code jointly to empower semantic search of codebases. 

It is comprised of a lightweight, async fastapi application running on a guicorn webserver. On startup, any of the supported models will be injected into the container, converted to an optimized serving format, and run on a REST API. 

CodeClarity automatically handles checking for supported languages for code models, dynamic batching of both code and natrual language snippits and conversions of code models to torchscript all in an asyncronous manner! 

#Getting started 
This repo impliments a docker container that serves the REST API. to automatically build the container with any of the supported models for code search by running the following 

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

## <a name="help"></a>Getting help

If you have any questions about, feedback for or a problem with Weave Net:

- Read [documatic website](https://www.documatic.com/).
- Sign up for the [documatic Waitlist](https://documatic-website.vercel.app/waitlist).
- [File an issue or request a feature](https://github.com/DocumaticAI/Roadmap).