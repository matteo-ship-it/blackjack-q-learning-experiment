As part of my Maturaarbeit, I explored reinforcement learning with a particular focus on tabular Q-learning. To understand how Q-learning works in practice, I implemented a Q-learning agent in Python for the simplified Blackjack-v1 environment from Gymnasium.

The program trains the agent with different training budgets and random seeds and compares its performance with a random policy and a reduced basic strategy. It automatically evaluates the policies, calculates summary statistics, and generates tables and figures used in the thesis.

The main implementation is located in `src/`, while the experiment and analysis scripts are located in `scripts/`.

# Reproducibility
The experiment can be reproduced by installing the dependencies listed in `requirements.txt` and running the scripts from the repository root:

```bash
pip install -r requirements.txt
python scripts/run_experiments.py
python scripts/create_experiment_summary.py
python scripts/create_results_table.py
python scripts/plot_mean_reward.py
python scripts/plot_policy_heatmaps.py
```

# Tests
The automated tests can be run with:

```bash
python -m unittest discover -s tests -v
```