# How to set up new instance:

## Get software

apt update -y && sudo apt upgrade -y && sudo apt-get install -y pkg-config python3-dev default-libmysqlclient-dev build-essential

## Get python libraries

pip install trl datasets torch deepspeed

## Clone repo

git clone https://github.com/Sam-de-Ham/finetuning-tests.git

## Other important directories and commands

/workspace/finetuning-tests/ds_config.json

/root/.cache/huggingface/accelerate/default_config.yaml

accelerate launch --config_file ./accelerate.yaml full_training_simple_sft.py
