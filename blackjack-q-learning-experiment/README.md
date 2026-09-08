As part of my Maturaarbeit, I explored reinforcement learning with a particular focus on tabular Q-learning. To understand how Q-learning works in practice, I implemented a Q-learning agent in Python for the simplified Blackjack-v1 environment from Gymnasium.

The program trains the agent with different training budgets and random seeds and compares its performance with a random policy and a reduced basic strategy. It automatically evaluates the policies, calculates summary statistics, and generates tables and figures used in the thesis.

The main implementation is located in `src/`, while the experiment and analysis scripts are located in `scripts/`.

The tests can be run with `python -m unittest discover -s tests -v`.
