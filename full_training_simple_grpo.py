from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate.utils import DummyOptim, DummyScheduler
from trl import GRPOConfig, GRPOTrainer
from model_to_use import model_name


def main():
    try:
        # Initialize accelerator with DeepSpeed plugin
        accelerator = Accelerator()

        # Load dataset
        dataset = load_dataset("trl-lib/tldr", split="train")
        dataset = dataset.select(range(100))

        # Load base model and tokenizer
        # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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

        def reward_len(completions, **kwargs):
            return [-abs(20 - len(completion)) for completion in completions]

        training_args = GRPOConfig(
            output_dir=f"{model_name}_grpo",
            logging_steps=10,
            fp16=False,
            bf16=True,
            deepspeed="/workspace/finetuning-tests/ds_config.json",  # Use relative path
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_len,
            args=training_args,
            train_dataset=dataset,
        )

        # Prepare the trainer with accelerator
        trainer = accelerator.prepare(trainer)

        # Start training
        trainer.train()

        # Save the final model
        if accelerator.is_main_process:
            trainer.save_model()
    except Exception as e:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
