# requires: datasets trl

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator

dataset = load_dataset("trl-lib/tldr", split="train")


training_args = SFTConfig(output_dir="Qwen-Distill-1.5B-GRPO")


trainer = Accelerator().prepare(
    SFTTrainer(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        train_dataset=dataset,
        args=training_args,
    )
)


trainer.train()
