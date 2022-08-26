
#!/bin/bash

if [ -f config.env ]; then
    echo "config.env already exists, skipping"
    echo "Please delete config.env if you want to re-run this script"
    exit 0
fi


UniXCoder:
  allowed_base_models: 
    microsoft/unixcoder-base : ["java", "ruby", "python", "php", "javascript", "go"] 
    microsoft/unixcoder-base-unimodal : ["java", "ruby", "python", "php", "javascript", "go"]
    microsoft/unixcoder-base-nine : ["java", "ruby", "python", "php", "javascript", "go", "c++", "c#", "c"]

  serving : 
    batch_size : 32 
    max_length : 512

echo "Models available:"
echo "[1] microsoft-unixcoder-base (1.5 total VRAM required; ["java", "ruby", "python", "php", "javascript", "go"] language support)"
echo "[2] microsoft-unixcoder-base-unimodal (2GB total VRAM required; ["java", "ruby", "python", "php", "javascript", "go"] language support)"
echo "[3] microsoft-unixcoder-base-nine (2GB total VRAM required; ["java", "ruby", "python", "php", "javascript", "go", "c++", "c#", "c"])"

# Read their choice
read -p "Enter your choice [4]: " MODEL_NUM

# Convert model number to model name
case $MODEL_NUM in
    1) MODEL="microsoft-unixcoder-base" ;;
    2) MODEL="microsoft-unixcoder-base-unimodal" ;;
    3) MODEL="microsoft-unixcoder-base-nine" ;;

esac


echo "Input the number of instances to run"

# Read their choice
read -p "Enter your choice [1]: " instance_num

export base_model=MODEL

docker build -t "codesearch_embeddings_api" .
#docker run -it codesearch_embeddings_api
