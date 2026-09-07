import gymnasium as gym
import numpy as np
from tqdm import tqdm

from .blackjack_agent import BlackjackAgent

def evaluate_policy(policy_function, number_of_episodes: int, seed: int) -> dict:
    """Evaluate a fixed policy over a given number of episodes."""
    env = gym.make("Blackjack-v1", natural=True, sab=False)
    env.action_space.seed(seed)

    episode_rewards = []
    wins = 0
    losses = 0
    draws = 0

    # tqdm displays a progress bar while the evaluation episodes run.
    for episode in tqdm(
        range(number_of_episodes),
        desc=f"Evaluating seed {seed}",
        leave=False,
    ):
        # Use a different environment seed for each evaluation episode.
        episode_seed = seed * number_of_episodes + episode
        obs, info = env.reset(seed=episode_seed)
        done = False
        episode_reward = 0

        while not done:
            # Select an action according to the policy being evaluated.
            action = policy_function(obs, env)

            obs, reward, terminated, truncated, info = env.step(action)
            reward = float(reward)

            episode_reward = episode_reward + reward
            done = terminated or truncated

        # Store the total reward from this episode.
        episode_rewards.append(episode_reward)

        # Count whether the episode was won, lost or drawn.
        if episode_reward > 0:
            wins = wins + 1
        elif episode_reward < 0:
            losses = losses + 1
        else:
            draws = draws + 1


    env.close()

    average_reward = sum(episode_rewards) / number_of_episodes

    return {
        "average_reward": average_reward,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }


def evaluate_trained_agent(
    agent: BlackjackAgent,
    number_of_episodes: int,
    seed: int,
) -> dict:
    """Evaluate a trained Q-learning agent using its greedy policy."""
    env = gym.make("Blackjack-v1", natural=True, sab=False)

    episode_rewards = []
    wins = 0
    losses = 0
    draws = 0

    # tqdm displays a progress bar while the evaluation episodes run.
    for episode in tqdm(
        range(number_of_episodes),
        desc=f"Evaluating trained agent seed {seed}",
        leave=False,
    ):
        # Use a different environment seed for each evaluation episode.
        episode_seed = seed * number_of_episodes + episode
        obs, info = env.reset(seed=episode_seed)
        done = False
        episode_reward = 0

        while not done:
            # Agent selects the action with the highest Q-value.
            action = int(np.argmax(agent.q_values[obs]))

            obs, reward, terminated, truncated, info = env.step(action)
            reward = float(reward)

            episode_reward = episode_reward + reward
            done = terminated or truncated

        # Store the total reward from this episode.
        episode_rewards.append(episode_reward)

        # Count whether the episode was won, lost or drawn.
        if episode_reward > 0:
            wins = wins + 1
        elif episode_reward < 0:
            losses = losses + 1
        else:
            draws = draws + 1

    env.close()

    average_reward = sum(episode_rewards) / number_of_episodes
    return {
        "average_reward": average_reward,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }