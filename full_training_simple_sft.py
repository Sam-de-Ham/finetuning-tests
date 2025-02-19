from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import set_seed
from accelerate.logging import get_logger
import torch
import os
import logging

# Initialize basic logging first
basic_logger = logging.getLogger(__name__)
basic_logger.setLevel(logging.DEBUG)

# Add at the top of the file
os.environ["NCCL_TIMEOUT"] = "120"  # 30 minutes instead of default 10
os.environ["NCCL_DEBUG"] = "INFO"  # Enable NCCL debugging


def main():
    try:
        # Initialize accelerator first
        fsdp_plugin = FullyShardedDataParallelPlugin(
            auto_wrap_policy={"transformer_layer_cls": "Qwen2DecoderLayer"},
            backward_prefetch="BACKWARD_PRE",
            state_dict_type="SHARDED_STATE_DICT",
            cpu_offload=False,  # Disable CPU offload to reduce communication overhead
            sync_module_states=True,
            param_init_fn=None,
        )

        basic_logger.debug("Initializing accelerator")
        accelerator = Accelerator(
            gradient_accumulation_steps=4,
            mixed_precision="bf16",  # Use bfloat16 for better memory efficiency
            fsdp_plugin=fsdp_plugin,
            log_with="all",  # Enable logging
        )

        # Now we can get and use the accelerate logger
        logger = get_logger(__name__, log_level="DEBUG")

        # Set random seed for reproducibility
        set_seed(42)

        # Load dataset
        logger.debug("Loading dataset")
        dataset = load_dataset("trl-lib/tldr", split="train")

        # Load base model and tokenizer
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        logger.debug("Loading model and tokenizer")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            use_cache=False,  # Disable KV cache for training
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Configure training arguments with more conservative settings
        training_args = SFTConfig(
            output_dir=f"{model_name}_SFT_tuned",
            max_seq_length=1024,  # Reduced from 2048
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,  # Increased from 4
            learning_rate=1e-5,  # Reduced from 2e-5
            num_train_epochs=3,
            fp16=False,
            bf16=True,
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            max_grad_norm=1.0,  # Add gradient clipping
        )

        # Initialize trainer with prepared model
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        # Prepare the trainer with accelerator
        trainer = accelerator.prepare(trainer)

        # Start training with error handling
        with accelerator.main_process_first():
            trainer.train()

        # Save the final model - only on the main process
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(trainer.model)
            accelerator.save(
                unwrapped_model.state_dict(),
                f"{training_args.output_dir}/final_model.pt",
            )

    except Exception as e:
        basic_logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
