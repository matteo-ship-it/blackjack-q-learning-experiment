import csv
import statistics
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data" / "final_experiment_results.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "final_experiment_summary.csv"


def read_results(filename: str | Path) -> list[dict]:
    """Read the experiment results from a CSV file."""
    with open(filename, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def create_summary(results: list[dict]) -> list[dict]:
    """Create summary statistics for each policy and training budget."""

    # Group rows by policy and training budget.
    grouped_results = defaultdict(list)

    for row in results:
        policy = row["policy"]
        training_budget = int(row["training_budget"])

        grouped_results[(policy, training_budget)].append(row)

    summary_rows = []

    # Go through each group.
    for (policy, training_budget), rows in grouped_results.items():
        average_rewards = []
        total_wins = 0
        total_losses = 0
        total_draws = 0

        for row in rows:
            average_rewards.append(float(row["average_reward"]))
            total_wins = total_wins + int(row["wins"])
            total_losses = total_losses + int(row["losses"])
            total_draws = total_draws + int(row["draws"])

        total_episodes = total_wins + total_losses + total_draws

        mean_reward = statistics.mean(average_rewards)
        standard_deviation = statistics.stdev(average_rewards)

        win_rate = total_wins / total_episodes
        loss_rate = total_losses / total_episodes
        draw_rate = total_draws / total_episodes

        summary_rows.append({
            "policy": policy,
            "training_budget": training_budget,
            "mean_reward": mean_reward,
            "standard_deviation": standard_deviation,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "total_draws": total_draws,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "draw_rate": draw_rate,
        })

    return summary_rows


def save_summary(summary_rows: list[dict], filename: str | Path):
    """Save the summary statistics to a CSV file."""
    fieldnames = [
        "policy",
        "training_budget",
        "mean_reward",
        "standard_deviation",
        "total_wins",
        "total_losses",
        "total_draws",
        "win_rate",
        "loss_rate",
        "draw_rate",
    ]

    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    results = read_results(INPUT_FILE)
    summary_rows = create_summary(results)
    save_summary(summary_rows, OUTPUT_FILE)

    print("Summary saved to", OUTPUT_FILE)