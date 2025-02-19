from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate.utils import DummyOptim, DummyScheduler


def print_total_parameters(model):
    """Count and print the total number of parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    total_params_millions = total_params / 1_000_000
    print(f"Total parameters: {total_params:,} (~{total_params_millions:.2f}M)")
    return total_params


def print_trainable_parameters(model):
    """Count and print trainable vs total parameters in the model."""
    trainable_params = 0
    all_params = 0

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_param_name = (
                f"{module_name}.{param_name}" if module_name else param_name
            )
            num_params = param.numel()
            all_params += num_params
            if param.requires_grad:
                print(f"Trainable layer: {full_param_name} ({num_params:,} parameters)")
                trainable_params += num_params

    trainable_percent = 100 * trainable_params / all_params if all_params > 0 else 0

    print(f"\nModel Summary:")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {all_params:,}")
    print(f"Trainable parameters percentage: {trainable_percent:.2f}%")

    return trainable_params, all_params


def main():
    # Initialize accelerator with DeepSpeed plugin
    accelerator = Accelerator()

    # Load dataset
    dataset = load_dataset("trl-lib/tldr", split="train")
    dataset = dataset.select(range(100))

    # Load base model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize model with no device map for DeepSpeed compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        torch_dtype=torch.bfloat16,
    )

    # Print initial parameter count
    total_params = print_total_parameters(model)

    # Print layer names if needed
    # for name, module in model.named_modules():
    #     print(f"name='{name}'")

    whitelist_layer_patterns = [
        "model.embed_tokens",  # Embedding layer - ALWAYS trainable - very big
        "model.layers.0",  # First Transformer Layer Block - KEEP trainable as requested
        "model.norm",  # Final LayerNorm - Often trainable
        "lm_head",  # Language Model Head - ALWAYS trainable - very big
    ]

    # Freeze all parameters first
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            param.requires_grad = False

    # Unfreeze only whitelisted layers
    unfrozen_count = 0
    for module_name, module in model.named_modules():
        if any(module_name.startswith(pattern) for pattern in whitelist_layer_patterns):
            for param_name, param in module.named_parameters(recurse=False):
                if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                    param.requires_grad = True
                    unfrozen_count += 1
                    print(f"Unfrozen layer: {module_name}.{param_name}")

    print(f"\nUnfrozen {unfrozen_count} parameter groups based on whitelist")

    # Print parameter statistics
    print("\nInitial parameter counts:")
    total_params = print_total_parameters(model)
    print("\nTrainable parameter analysis:")
    trainable_params, all_params = print_trainable_parameters(model)

    # Configure training arguments with absolute path to DeepSpeed config
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
        deepspeed="/workspace/finetuning-tests/ds_config.json",  # Use relative path
    )

    # Initialize trainer with accelerator
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Prepare the trainer with accelerator
    trainer = accelerator.prepare(trainer)

    # Start training
    trainer.train()

    # Save the final model
    if accelerator.is_main_process:
        trainer.save_model()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(traceback.format_exc())
