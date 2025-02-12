# requires: datasets trl torch

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")

from trl import GRPOConfig, GRPOTrainer


def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


training_args = GRPOConfig(output_dir="Qwen-Distill-1.5B-GRPO", logging_steps=10)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)

trainer.model.device_map = "auto"


for name, module in trainer.model.named_modules():
    print(f"name='{name}'")


def print_total_parameters(model):
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
    print(f"Total parameters in the original model: {all_param}")


print_total_parameters(trainer.model)


import torch

whitelist_layer_patterns = [
    "model.embed_tokens",  # Embedding layer - ALWAYS trainable - very big
    "model.layers.0",  # First Transformer Layer Block - KEEP trainable as requested
    "model.norm",  # Final LayerNorm - Often trainable
    "lm_head",  # Language Model Head - ALWAYS trainable - very big
]


# Freeze all parameters EXCEPT those in the whitelist
for n, p in trainer.model.named_parameters():
    p.requires_grad = False  # Freeze all layers by default
    for layer_pattern in whitelist_layer_patterns:
        if (
            layer_pattern in n
        ):  # Check if the current layer name matches any whitelist pattern
            if p.dtype in [
                torch.float16,
                torch.float32,
                torch.bfloat16,
                torch.complex64,
                torch.complex128,
            ]:  # ADDED dtype check!
                p.requires_grad = (
                    True  # Unfreeze if it's in the whitelist AND it's a float type
                )
            break  # No need to check other patterns if already whitelisted


# Verification - Count trainable parameters. Should be significantly less than full model.
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


print_trainable_parameters(trainer.model)


trainer.train()
