# requires: datasets trl

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate.utils import InitProcessGroupKwargs
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
import torch

# Load dataset
dataset = load_dataset("trl-lib/tldr", split="train")

# Load base model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=None,  # Required for FSDP
    torch_dtype=torch.bfloat16,  # Use bfloat16 for better training stability
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure training arguments
training_args = SFTConfig(
    output_dir=f"{model_name}_finetuned",
    max_seq_length=2048,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=False,
    bf16=True,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
)

# Initialize trainer with accelerator
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

# Start training
trainer.train()
