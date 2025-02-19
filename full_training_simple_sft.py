from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import set_seed, FullyShardedDataParallelPlugin
import torch


def main():
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        fsdp_plugin=FullyShardedDataParallelPlugin(
            auto_wrap_policy={"transformer_layer_cls": "Qwen2DecoderLayer"},
            backward_prefetch="BACKWARD_PRE",
            state_dict_type="SHARDED_STATE_DICT",
            cpu_offload=True,
            sync_module_states=True,
            param_init_fn=None,
        ),
    )

    # Set random seed for reproducibility
    set_seed(42)

    # Load dataset
    dataset = load_dataset("trl-lib/tldr", split="train")

    # Load base model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Important for training with FSDP
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
        gradient_checkpointing=True,
    )

    # Prepare model with accelerator
    model = accelerator.prepare(model)

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # Start training
    trainer.train()

    # Save the final model
    trainer.save_model()


if __name__ == "__main__":
    main()
