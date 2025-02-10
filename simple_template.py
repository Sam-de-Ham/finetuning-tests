from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="Qwen-Distill-1.5B-GRPO", logging_steps=10)
trainer = GRPOTrainer(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()