import unittest

import gymnasium as gym
import numpy as np

from src.blackjack_agent import BlackjackAgent


class TestBlackjackAgent(unittest.TestCase):
    def setUp(self):
        self.env = gym.make("Blackjack-v1")

        self.agent = BlackjackAgent(
            env=self.env,
            learning_rate=0.5,
            initial_epsilon=0.0,
            epsilon_decay=0.1,
            final_epsilon=0.1,
            discount_factor=1.0,
        )

    def tearDown(self):
        self.env.close()

    def test_greedy_action_uses_highest_q_value(self):
        state = (16, 10, False)
        self.agent.q_values[state] = np.array([1.0, 3.0])

        action = self.agent.get_action(state)

        self.assertEqual(action, 1)

    def test_terminal_update_uses_reward(self):
        state = (16, 10, False)
        self.agent.update(
            obs=state,
            action=1,
            reward=1.0,
            terminated=True,
            next_obs=state,
        )

        self.assertEqual(self.agent.q_values[state][1], 0.5)

    def test_epsilon_does_not_drop_below_final_value(self):
        self.agent.epsilon = 0.15
        self.agent.decay_epsilon()

        self.assertEqual(self.agent.epsilon, 0.1)


if __name__ == "__main__":
    unittest.main()