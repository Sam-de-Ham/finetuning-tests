# requires: datasets trl

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from accelerate import Accelerator

dataset = load_dataset("trl-lib/tldr", split="train")


# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]


training_args = SFTConfig(output_dir="Qwen-Distill-1.5B-GRPO")
# trainer = GRPOTrainer(
#     model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#     reward_funcs=reward_len,
#     args=training_args,
#     train_dataset=dataset,
# )

trainer = SFTTrainer(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    train_dataset=dataset,
    args=training_args,
)

trainer.model = accelerator.prepare(trainer.model)

trainer.train()
