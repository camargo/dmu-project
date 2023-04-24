from aim import Run
import gymnasium as gym
from helpers import save_model
from statistics import mean
import torch


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
    return torch.from_numpy(I.ravel()).float()


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


def main():
    aim_run = Run()
    aim_run_id = aim_run.name.split(" ").pop()

    # Hyperparameters
    in_dim = 6400
    hidden_dim = 200
    out_dim = 2
    gamma = 0.99
    learning_rate = 1e-4
    aim_run["hparams"] = {
        "gamma": gamma,
        "hidden_dim": hidden_dim,
        "in_dim": in_dim,
        "learning_rate": learning_rate,
        "out_dim": out_dim,
    }

    # Initialization
    policy = PongPolicy(in_dim, hidden_dim, out_dim)
    env = gym.make("ALE/Pong-v5")
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    for episode in range(1, 20_000):
        policy.reset()
        observation, _ = env.reset()

        for _ in range(1, 1_000):
            observation = pong_observation(observation)
            action = policy.action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            policy.rewards.append(reward)

            if terminated or truncated:
                break

        loss = train(policy, optimizer, gamma)
        avg_reward = mean(policy.rewards)
        total_reward = sum(policy.rewards)

        frame_number = info.get("frame_number")
        aim_run.track(loss, "loss", frame_number, episode)
        aim_run.track(avg_reward, "avg_reward", frame_number, episode)
        aim_run.track(total_reward, "total_reward", frame_number, episode)

        save_model(policy, aim_run_id)

    env.close()


if __name__ == "__main__":
    main()
