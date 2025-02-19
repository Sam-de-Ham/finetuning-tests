from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate.utils import DummyOptim, DummyScheduler


def main():
    # Initialize accelerator with DeepSpeed plugin
    accelerator = Accelerator()

    # Load dataset
    dataset = load_dataset("trl-lib/tldr", split="train")

    # Load base model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize model with no device map for DeepSpeed compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        torch_dtype=torch.bfloat16,
    )

    for name, module in model.named_modules():
        print(f"name='{name}'")

    def print_total_parameters(model):
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        print(f"Total parameters in the original model: {all_param}")

    print_total_parameters(model)

    whitelist_layer_patterns = [
        "model.embed_tokens",  # Embedding layer - ALWAYS trainable - very big
        "model.layers.0",  # First Transformer Layer Block - KEEP trainable as requested
        "model.norm",  # Final LayerNorm - Often trainable
        "lm_head",  # Language Model Head - ALWAYS trainable - very big
    ]

    # Freeze all parameters EXCEPT those in the whitelist
    for n, p in model.named_parameters():
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

    print_trainable_parameters(model)

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
    main()
