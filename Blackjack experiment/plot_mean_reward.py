import csv
from pathlib import Path

import matplotlib.pyplot as plt


# This is the summary file created from the raw experiment results.
SUMMARY_FILE = "final_experiment_summary.csv"

# This folder will store the finished figure.
OUTPUT_FOLDER = Path("figures")


def read_summary(filename: str) -> list[dict]:
    """Read the summary CSV file."""

    with open(filename, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def get_row(summary_rows: list[dict], policy: str, training_budget: int) -> dict:
    """Find one specific row in the summary table."""

    for row in summary_rows:
        same_policy = row["policy"] == policy
        same_budget = int(row["training_budget"]) == training_budget

        if same_policy and same_budget:
            return row

    raise ValueError(f"Could not find row for {policy}, budget {training_budget}")


def get_q_learning_rows(summary_rows: list[dict]) -> list[dict]:
    """Get only the Q-learning rows and sort them by training budget."""

    q_learning_rows = []

    for row in summary_rows:
        if row["policy"] == "Q-learning agent":
            q_learning_rows.append(row)

    q_learning_rows.sort(key=lambda row: int(row["training_budget"]))

    return q_learning_rows


def format_training_budget(training_budget: int) -> str:
    """Convert large training budget numbers into shorter labels."""

    if training_budget == 1_000_000:
        return "1M"

    if training_budget >= 1_000:
        return f"{training_budget // 1_000}k"

    return str(training_budget)


def plot_mean_reward(summary_rows: list[dict]):
    """Create the mean reward plot for the Q-learning agent."""

    # Select the Q-learning rows from the summary file.
    q_learning_rows = get_q_learning_rows(summary_rows)

    # Create x-positions for the five Q-learning training budgets.
    x_positions = list(range(len(q_learning_rows)))

    # Read training budgets, mean rewards and standard deviations.
    training_budgets = []
    mean_rewards = []
    standard_deviations = []

    for row in q_learning_rows:
        training_budgets.append(int(row["training_budget"]))
        mean_rewards.append(float(row["mean_reward"]))
        standard_deviations.append(float(row["standard_deviation"]))

    # Create readable x-axis labels such as 10k, 50k and 1M.
    x_labels = []

    for training_budget in training_budgets:
        x_labels.append(format_training_budget(training_budget))

    # Get the baseline rewards.
    basic_strategy_row = get_row(summary_rows, "Reduced basic strategy", 0)
    basic_strategy_reward = float(basic_strategy_row["mean_reward"])

    # Create the figure.
    plt.figure(figsize=(8, 5))

    # Plot the Q-learning mean rewards.
    # The error bars show the standard deviation across the 10 seeds.
    plt.errorbar(
        x_positions,
        mean_rewards,
        yerr=standard_deviations,
        marker="o",
        capsize=5,
        label="Q-learning agent",
    )

    # Add horizontal reference lines for the two baseline policies.
    plt.axhline(
        basic_strategy_reward,
        linestyle="--",
        label="Reduced basic strategy",
    )


    # Add labels and title.
    plt.xticks(x_positions, x_labels)
    plt.xlabel("Training episodes")
    plt.ylabel("Mean evaluation reward")
    plt.title("Q-learning performance by training budget")

    # Add grid and legend.
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    # Make sure the labels fit nicely into the image.
    plt.tight_layout()

    # Save the figure as a high-resolution PNG file.
    plt.savefig(
        OUTPUT_FOLDER / "q_learning_mean_reward_by_training_budget.png",
        dpi=300,
    )

    plt.close()


if __name__ == "__main__":
    # Create the figures folder if it does not already exist.
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    # Read the summary data.
    summary_rows = read_summary(SUMMARY_FILE)

    # Create and save the plot.
    plot_mean_reward(summary_rows)

    print("Figure saved in the figures folder.")