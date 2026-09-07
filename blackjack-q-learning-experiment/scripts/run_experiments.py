import csv
import statistics
from pathlib import Path

from src.evaluation import evaluate_policy, evaluate_trained_agent
from src.policies import random_policy, reduced_basic_strategy
from src.training import train_q_learning_agent


TRAINING_BUDGETS = [10_000, 50_000, 100_000, 500_000, 1_000_000]
SEEDS = range(10)
EVALUATION_EPISODES = 100_000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "final_experiment_results.csv"




def run_baseline_experiments(results):
    """Evaluate the random policy and the reduced basic strategy."""

    # Store policy names together with their corresponding functions.
    baseline_policies = {
        "Random policy": random_policy,
        "Reduced basic strategy": reduced_basic_strategy,
    }

    # Evaluate each baseline policy separately.
    for policy_name, policy_function in baseline_policies.items():
        average_rewards = []

        print("Evaluating:", policy_name)

        for seed in SEEDS:
            # Evaluate the current policy with one random seed.
            evaluation_result = evaluate_policy(
                policy_function=policy_function,
                number_of_episodes=EVALUATION_EPISODES,
                seed=100 +seed,
            )

            average_reward = evaluation_result["average_reward"]

            average_rewards.append(average_reward)
            results.append({
                "policy": policy_name,
                "training_budget": 0,
                "seed": 100 + seed,
                "average_reward": average_reward,
                "wins": evaluation_result["wins"],
                "losses": evaluation_result["losses"],
                "draws": evaluation_result["draws"],
            })

            print("Seed:", seed, "Average reward:", average_reward)

        # Calculate summary statistics over all seeds.
        mean_reward = statistics.mean(average_rewards)
        standard_deviation = statistics.stdev(average_rewards)

        print("Policy:", policy_name)
        print("Mean reward:", mean_reward)
        print("Standard deviation:", standard_deviation)
        print("---")


def run_q_learning_experiments(results):
    """Train and evaluate Q-learning agents for all training budgets."""

    for training_budget in TRAINING_BUDGETS:
        average_rewards = []

        print("Training budget:", training_budget)

        for seed in SEEDS:
            # Train one agent for the current budget and seed.
            agent = train_q_learning_agent(
                n_episodes=training_budget,
                seed=seed,
            )

            # Evaluate the trained agent using its greedy policy.
            evaluation_result = evaluate_trained_agent(
                agent=agent,
                number_of_episodes=EVALUATION_EPISODES,
                seed=100 + seed,
            )

            average_reward = evaluation_result["average_reward"]

            average_rewards.append(average_reward)
            results.append({
                "policy": "Q-learning agent",
                "training_budget": training_budget,
                "seed": 100 + seed,
                "average_reward": average_reward,
                "wins": evaluation_result["wins"],
                "losses": evaluation_result["losses"],
                "draws": evaluation_result["draws"],
            })

            print("Seed:", seed, "Average reward:", average_reward)

        # Calculate summary statistics over all seeds for this training budget.
        mean_reward = statistics.mean(average_rewards)
        standard_deviation = statistics.stdev(average_rewards)

        print("Training budget:", training_budget)
        print("Mean reward:", mean_reward)
        print("Standard deviation:", standard_deviation)
        print("---")


def save_results_to_csv(results, filename: str | Path):
    """Save experiment results to a CSV file."""
    fieldnames = [
        "policy",
        "training_budget",
        "seed",
        "average_reward",
        "wins",
        "losses",
        "draws",
    ]

    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    results = []

    print("Starting baseline experiments")
    print("---")
    run_baseline_experiments(results)

    print("Starting Q-learning experiments")
    print("---")
    run_q_learning_experiments(results)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    save_results_to_csv(results, OUTPUT_FILE)
    print(f"Results saved to {OUTPUT_FILE}")
