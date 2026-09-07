from collections import defaultdict

import gymnasium as gym
import numpy as np

class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 1.0,
    ):
        """Create a Q-learning agent for the Blackjack environment."""
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(2))  # Creates the Q-table

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Select an action using an epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return int(np.argmax(self.q_values[obs]))  # Exploit

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update the Q-values based on the observed transition."""
        # If the episode ended, there is no future Q-value.
        if terminated:
            future_q_value = 0.0
        else: future_q_value = np.max(self.q_values[next_obs])

        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce epsilon after each training episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

