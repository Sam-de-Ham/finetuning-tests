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
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.qwen.modeling_qwen import QWenBlock
import torch
import datetime

# Initialize FSDP Plugin with proper configuration
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    ),
    auto_wrap_policy=transformer_auto_wrap_policy(transformer_layer_cls={QWenBlock}),
)

# Initialize accelerator with FSDP settings
accelerator = Accelerator(
    gradient_accumulation_steps=4,
    mixed_precision="bf16",  # Use mixed precision for better memory efficiency
    fsdp_plugin=fsdp_plugin,
)

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
    output_dir="Qwen-Distill-1.5B-GRPO",
    max_seq_length=2048,
    per_device_train_batch_size=1,  # Reduced batch size as model is sharded
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=False,  # We're using bf16 instead
    bf16=True,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
)

# Initialize trainer with accelerator
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    accelerator=accelerator,
)

# Start training
trainer.train()
