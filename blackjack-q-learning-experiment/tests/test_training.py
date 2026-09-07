import unittest

from src.training import train_q_learning_agent


class TestTraining(unittest.TestCase):
    def test_training_returns_agent_with_learned_values(self):
        agent = train_q_learning_agent(
            n_episodes=10,
            seed=0,
        )

        self.assertGreater(len(agent.q_values), 0)
        self.assertGreaterEqual(agent.epsilon, agent.final_epsilon)
        self.assertLessEqual(agent.epsilon, 1.0)


if __name__ == "__main__":
    unittest.main()