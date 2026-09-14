import unittest

from src.policies import HIT, STICK, reduced_basic_strategy


class TestReducedBasicStrategy(unittest.TestCase):
    def test_hard_hand_hits_below_threshold(self):
        action = reduced_basic_strategy(
            obs=(16, 10, False),
            env=None,
        )

        self.assertEqual(action, HIT)

    def test_hard_hand_stands_at_threshold(self):
        action = reduced_basic_strategy(
            obs=(17, 10, False),
            env=None,
        )

        self.assertEqual(action, STICK)

    def test_soft_hand_hits_below_threshold(self):
        action = reduced_basic_strategy(
            obs=(17, 1, True),
            env=None,
        )

        self.assertEqual(action, HIT)

    def test_soft_hand_stands_at_threshold(self):
        action = reduced_basic_strategy(
            obs=(18, 1, True),
            env=None,
        )

        self.assertEqual(action, STICK)


if __name__ == "__main__":
    unittest.main()