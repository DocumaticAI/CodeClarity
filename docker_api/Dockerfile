FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub


ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    nginx \ 
    apt-utils  \ 
    build-essential \
    net-tools \ 
    curl \ 
    wget \
 && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda install -y python==3.8.3 \
    && conda clean -ya

COPY requirements.txt .

# CUDA 11.0-specific steps
RUN conda install pytorch cudatoolkit=11.3 -c pytorch \
    && pip install -r requirements.txt \
    && pip install transformers -U \
    && conda clean -ya


COPY app /opt/program
WORKDIR /opt/program

RUN sudo chmod +x /opt/program/serve

RUN /opt/program/serve
