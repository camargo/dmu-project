import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from model_helpers import load_model
from pong_train import PongPolicy, pong_observation
from torch import manual_seed


def main():
    policy: PongPolicy = load_model("2b9c7df2eca04bb49e31404f")
    policy.reset()
    record_video = False
    use_seed = False

    if record_video:
        env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
        env = RecordVideo(env, "videos", name_prefix="pong-agent")
        env.metadata["render_fps"] = 30
    else:
        env = gym.make("ALE/Pong-v5")

    observation, _ = env.reset()

    if use_seed:
        seed = 3
        manual_seed(seed)
        env.seed(seed)

    total_agent_wins = 0
    total_computer_wins = 0
    total_games = 0
    agent_score = 0
    computer_score = 0

    for _ in range(1, 1_000_000):
        observation = pong_observation(observation)
        action = policy.action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        if reward > 0:
            agent_score += 1
        elif reward < 0:
            computer_score += 1

        if terminated or truncated:
            if record_video:
                break

            total_games += 1

            if agent_score == 21:
                total_agent_wins += 1
            elif computer_score == 21:
                total_computer_wins += 1

            print(f"{info=}")
            print(f"Agent score: {agent_score}, Computer score: {computer_score}")
            print(f"Total games: {total_games}")
            print(f"Total agent wins: {total_agent_wins}, Total computer wins: {total_computer_wins}")

            if total_games % 100 == 0:
                agent_percent_wins = (total_agent_wins / total_games) * 100
                computer_percent_wins = (total_computer_wins / total_games) * 100
                print(f"Agent percent wins: {agent_percent_wins:.2f}%")
                print(f"Computer percent wins: {computer_percent_wins:.2f}%")

            agent_score = 0
            computer_score = 0

            env.reset()
            policy.reset()

    env.close()


if __name__ == "__main__":
    main()
