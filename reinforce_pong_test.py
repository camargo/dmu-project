import gymnasium as gym
from reinforce_pong_train import PongPolicy, load_model, pong_observation


def main():
    model, _ = load_model("cf5e128fcf9543f58a5aab14744c733f")
    policy: PongPolicy = model
    policy.reset()

    env = gym.make("ALE/Pong-v5", render_mode="human")
    observation, _ = env.reset()

    for _ in range(20_000):
        observation = pong_observation(observation)
        action = policy.action(observation)
        observation, _, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
