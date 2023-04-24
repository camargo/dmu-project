import gymnasium as gym
from reinforce_pong_train import PongPolicy, load_model, pong_observation


def main():
    model, _ = load_model("cf5e128fcf9543f58a5aab14744c733f")
    policy: PongPolicy = model
    policy.reset()

    env = gym.make("ALE/Pong-v5", render_mode="human")
    observation, _ = env.reset()

    total_reward = 0
    total_games = 0

    for _ in range(50_000):
        observation = pong_observation(observation)
        action = policy.action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            total_games += 1

            print(f"{info=}")
            print(f"{total_reward=}")
            print(f"{total_games=}")

            total_reward = 0
            env.reset()
            policy.reset()

    env.close()


if __name__ == "__main__":
    main()
