import gymnasium as gym
import numpy as np
import random
import statistics
from torch import nn
import matplotlib.pyplot as plt

# from gymnasium.utils.play import play

env = gym.make("ALE/Pong-v5", render_mode="human")

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()

random.seed(1337)
env.seed(1337)
env.action_space.seed(1337)

observation, info = env.reset()

print(np.shape(observation))  # (210, 160, 3)

rewards = []

for i in range(10000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)

    if terminated or truncated:
        print(f"{i} terminated: {terminated}")  # Max score, 21
        print(f"{i} truncated: {truncated}")
        observation, info = env.reset()

print(f"mean reward: {statistics.mean(rewards)}")

env.close()
