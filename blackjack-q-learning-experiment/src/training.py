import gymnasium as gym
import numpy as np
from tqdm import tqdm

from .blackjack_agent import BlackjackAgent


LEARNING_RATE = 0.01
START_EPSILON = 1.0
FINAL_EPSILON = 0.1
DISCOUNT_FACTOR = 1.0


def train_q_learning_agent(n_episodes: int, seed: int) -> BlackjackAgent:
    """Train one Q-learning agent for a given number of episodes."""
    env = gym.make("Blackjack-v1", natural=True, sab=False)

    # Seed NumPy and the action space to make exploration reproducible.
    np.random.seed(seed)
    env.action_space.seed(seed)

    # Epsilon is reduced linearly during the first half of training.
    epsilon_decay = (START_EPSILON - FINAL_EPSILON) / (n_episodes / 2)

    agent = BlackjackAgent(
        env=env,
        learning_rate=LEARNING_RATE,
        initial_epsilon=START_EPSILON,
        epsilon_decay=epsilon_decay,
        final_epsilon=FINAL_EPSILON,
        discount_factor=DISCOUNT_FACTOR,
    )

    # tqdm displays a progress bar while the training episodes run.
    for episode in tqdm(
        range(n_episodes),
        desc=f"Training seed {seed}, budget {n_episodes}",
        leave=False,
    ):
        # Use a different environment seed for each training episode.
        episode_seed = seed * n_episodes + episode
        obs, info = env.reset(seed=episode_seed)
        done = False

        while not done:
            # Choose an action using the current epsilon-greedy policy.
            action = agent.get_action(obs)

            # Apply the action in the environment.
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward = float(reward)

            # Update the Q-table using the observed transition.
            agent.update(
                obs=obs,
                action=action,
                reward=reward,
                terminated=terminated,
                next_obs=next_obs,
            )

            done = terminated or truncated
            obs = next_obs

        # Reduce exploration after each episode.
        agent.decay_epsilon()

    env.close()
    return agent