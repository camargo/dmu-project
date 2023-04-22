import gymnasium as gym
import json
import numpy as np
import os
from time import time
import torch
import uuid


class PongPolicy(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super(PongPolicy, self).__init__()
        self.layer1 = torch.nn.Linear(in_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(hidden_dim, out_dim)
        self.softmax = torch.nn.Softmax(dim=0)
        self.reset()
        self.train()  # Set the module to training mode

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.softmax(x)
        return x

    def reset(self):
        self.log_probs: list[float] = []
        self.rewards: list[float] = []

    def action(self, x: torch.Tensor):
        action_probs = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)
        sample = dist.sample()
        log_prob = dist.log_prob(sample)
        self.log_probs.append(log_prob)
        actions = torch.tensor([2, 3])  # Right (2), Left (3)
        action = actions[sample]
        return action.item()


def pong_observation(observation):
    """
    Converts 3D (210, 160, 3) uint8 tensor into 1D (6400,) float vector for Pong.
    See: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5#file-pg-pong-py-L30
    """
    I = observation[35:195]  # Crop just game area (remove score)
    I = I[::2, ::2, 0]  # Down-sample by factor of 2
    I[I == 144] = 0  # Erase background (background type 1)
    I[I == 109] = 0  # Erase background (background type 2)
    I[I != 0] = 1  # Everything else (paddles, ball) just set to 1
    return torch.from_numpy(I.astype(float).ravel()).float()


def train(policy: PongPolicy, optimizer: torch.optim.Optimizer, gamma: float) -> float:
    reward_count = len(policy.rewards)
    rewards_to_go = torch.empty(reward_count, dtype=float)
    reward_to_go = 0.0

    for t in reversed(range(reward_count)):
        reward_to_go = policy.rewards[t] + gamma * reward_to_go
        rewards_to_go[t] = reward_to_go

    log_probs = torch.stack(policy.log_probs)
    loss = -log_probs * rewards_to_go
    loss = torch.sum(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def save_model(model: torch.nn.Module, run_id: str, metadata):
    """
    Save a model and associated metadata.
    """

    models_dir = ".models"

    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    model_run_dir = f"{models_dir}/{run_id}"

    if not os.path.exists(model_run_dir):
        os.mkdir(model_run_dir)

    torch.save(model, f"{model_run_dir}/model.pt")

    with open(f"{model_run_dir}/model.json", "w") as f:
        json.dump(metadata, f, indent=None)


def load_model(run_id: str):
    """
    Load a model and associated metadata.
    """

    model_run_dir = f".models/{run_id}"

    model: torch.nn.Module = torch.load(f"{model_run_dir}/model.pt")
    model.eval()

    with open(f"{model_run_dir}/model.json", "r") as f:
        metadata = json.load(f)

    return model, metadata


def main():
    resume = False
    run_id = uuid.uuid4().hex

    # resume = True
    # run_id = "aea4bb2b696a41c5b2937992ab8fa4a9"

    if resume:
        model, metadata = load_model(run_id)
        policy: PongPolicy = model
        policy.reset()
    else:
        policy = PongPolicy(6400, 200, 2)
        metadata = {}

    # Hyperparameters
    gamma = metadata.get("gamma", 0.99)
    learning_rate = metadata.get("learning_rate", 1e-4)

    # Initialization
    env = gym.make("ALE/Pong-v5", render_mode="human")
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    # Seeds
    # torch.manual_seed(1337)
    # env.seed(1337)

    # Bookkeeping
    start_time = metadata.get("start_time", time())
    episode_number = metadata["episode_number"] + 1 if "episode_number" in metadata else 1
    step = metadata["step"] + 1 if "step" in metadata else 1
    losses = metadata.get("losses", [])
    total_rewards = metadata.get("total_rewards", [])
    elapsed_times = metadata.get("elapsed_times", [])

    observation, _ = env.reset()

    for _ in range(1_000_000):
        observation = pong_observation(observation)
        action = policy.action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        policy.rewards.append(reward)

        if reward != 0 or terminated or truncated:
            loss = train(policy, optimizer, gamma)
            losses.append(loss)
            total_rewards.append(sum(policy.rewards))
            elapsed_times.append(time() - start_time)

            print(f"{step}: episode {episode_number}, loss: {loss}")
            save_model(
                policy,
                run_id,
                {
                    "elapsed_times": elapsed_times,
                    "episode_number": episode_number,
                    "gamma": gamma,
                    "learning_rate": learning_rate,
                    "losses": losses,
                    "run_id": run_id,
                    "start_time": start_time,
                    "step": step,
                    "total_rewards": total_rewards,
                },
            )

            policy.reset()
            observation, _ = env.reset()
            episode_number += 1

        step += 1


if __name__ == "__main__":
    main()
