from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils import set_seed
import torch


def main():
    # Initialize accelerator with FSDP plugin for memory-efficient multi-GPU training
    fsdp_plugin = FullyShardedDataParallelPlugin(
        auto_wrap_policy={"transformer_layer_cls": "Qwen2DecoderLayer"},
        backward_prefetch="BACKWARD_PRE",
        state_dict_type="SHARDED_STATE_DICT",
        cpu_offload=True,  # Offload parameters to CPU when not in use
        sync_module_states=True,
        param_init_fn=None,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",  # Use bfloat16 for better memory efficiency
        fsdp_plugin=fsdp_plugin,
    )

    # Set random seed for reproducibility
    set_seed(42)

    # Load dataset
    dataset = load_dataset("trl-lib/tldr", split="train")

    # Load base model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # Initialize model with mixed precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Disable KV cache for training
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure training arguments
    training_args = SFTConfig(
        output_dir=f"{model_name}_SFT_tuned",
        max_seq_length=2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=False,
        bf16=True,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
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

    # Start training
    with accelerator.main_process_first():
        trainer.train()

    # Save the final model - only on the main process
    if accelerator.is_main_process:
        # Get unwrapped model
        unwrapped_model = accelerator.unwrap_model(trainer.model)
        # Save using accelerator's save function
        accelerator.save(
            unwrapped_model.state_dict(), f"{training_args.output_dir}/final_model.pt"
        )


if __name__ == "__main__":
    main()
