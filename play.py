import gymnasium as gym
from gymnasium.utils.play import play

# https://www.gymlibrary.dev/environments/atari/complete_list/

game = "ALE/Asteroids-v5"
env = gym.make(game, render_mode="rgb_array", frameskip=1)
env.metadata["render_fps"] = 60
play(env, zoom=5)
