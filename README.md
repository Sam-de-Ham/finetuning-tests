# How to run

## Install requirements and clone repo

`apt update -y && apt upgrade -y && apt-get install -y pkg-config python3-dev default-libmysqlclient-dev build-essential`

`pip install trl datasets torch deepspeed`

`git clone https://github.com/Sam-de-Ham/finetuning-tests.git`

## Start a run

Use accelerate to run scripts to ensure proper multi-GPU usage

Config for accelerate is passed as parameter, which then references the DeepSpeed config

`accelerate launch --config_file ./accelerate.yaml full_training_simple_sft.py`

## Important settings

`accelerate.yaml` - settings for accelerate library to configure multi-GPU behavior

| Parameter       | Description                     | Example Value |
| --------------- | ------------------------------- | ------------- |
| `num_machines`  | Number of machines for training | `1`           |
| `num_processes` | Number of GPUs for this machine | `4`           |

`ds_config.json` - settings for DeepSpeed library to configure multi-GPU behavior

| Parameter                  | Description                            | Example Value |
| -------------------------- | -------------------------------------- | ------------- |
| `offload_optimizer/device` | device to use for optimizer offloading | `none`/ `cpu` |
| `offload_param/device`     | device to use for parameter offloading | `none`/ `cpu` |

# Libraries used

### [Accelerate](https://github.com/huggingface/accelerate) - [documentation](https://huggingface.co/docs/accelerate/en/index) - multi-GPU training

### [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) - [documentation](https://deepspeed.readthedocs.io/en/latest/) - model sharding

### [trl](https://github.com/huggingface/trl) - [documentation](https://huggingface.co/docs/trl/index) - transformer training

### [transformers](https://github.com/huggingface/transformers) - [documentation](https://huggingface.co/docs/transformers/index) - model management

# Links

GRPO training information from trl - [link](https://huggingface.co/docs/trl/main/en/grpo_trainer)

# Data collected

| Model       | Size | Type of training | Vram (GB) |
| ----------- | ---- | ---------------- | --------- |
| DeepSeek-R1 | 1.5B | SFT              | > 32      |
| DeepSeek-R1 | 1.5B | GRPO             | 90-100    |
