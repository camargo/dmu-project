import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from model_helpers import load_model
from pong_train import PongPolicy, pong_observation


def main():
    policy: PongPolicy = load_model("2b9c7df2eca04bb49e31404f")
    policy.reset()
    record_video = False

    if record_video:
        env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        env = RecordVideo(env, "videos", name_prefix="pong-agent")
        env.metadata["render_fps"] = 30
    else:
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
            if record_video:
                break

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
