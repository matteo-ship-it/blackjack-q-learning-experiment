import unittest

from src.evaluation import evaluate_policy
from src.policies import random_policy


class TestEvaluation(unittest.TestCase):
    def test_evaluation_returns_complete_results(self):
        result = evaluate_policy(
            policy_function=random_policy,
            number_of_episodes=10,
            seed=0,
        )

        self.assertIn("average_reward", result)
        self.assertIn("wins", result)
        self.assertIn("losses", result)
        self.assertIn("draws", result)

        total_outcomes = (
            result["wins"]
            + result["losses"]
            + result["draws"]
        )

        self.assertEqual(total_outcomes, 10)


if __name__ == "__main__":
    unittest.main()