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
    dataset = dataset.select(range(100))

    # Load base model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # Initialize model with no device map for DeepSpeed compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
