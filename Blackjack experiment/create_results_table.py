import csv


INPUT_FILE = "results/final_experiment_summary.csv"
OUTPUT_FILE = "results/results_table_for_word.csv"


def read_summary(filename: str) -> list[dict]:
    """Read the summary CSV file."""

    with open(filename, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def format_training_budget(training_budget: int) -> str:
    """Format the training budget for the Word table."""

    # Baseline policies are not trained, so we show a dash.
    if training_budget == 0:
        return "-"

    # Add commas to large numbers, for example 1000000 -> 1,000,000.
    return f"{training_budget:,}"


def get_rate_percent(row: dict, rate_name: str) -> float:
    """
    Read a win/loss/draw rate and return it as a percentage.

    This works for both possible summary formats:
    win_rate = 0.428        -> 42.8
    win_rate_percent = 42.8 -> 42.8
    """

    percent_column = rate_name + "_percent"

    if percent_column in row:
        return float(row[percent_column])

    return float(row[rate_name]) * 100


def find_row(summary_rows: list[dict], policy: str, training_budget: int) -> dict:
    """Find one specific row in the summary table."""

    for row in summary_rows:
        same_policy = row["policy"] == policy
        same_budget = int(row["training_budget"]) == training_budget

        if same_policy and same_budget:
            return row

    raise ValueError(f"Could not find row for {policy}, budget {training_budget}")


def create_table_rows(summary_rows: list[dict]) -> list[dict]:
    """Create rounded table rows for the Maturaarbeit."""

    table_rows = []

    # This defines the order in which the rows should appear in the final table.
    row_order = [
        ("Random policy", 0),
        ("Reduced basic strategy", 0),
        ("Q-learning agent", 10_000),
        ("Q-learning agent", 50_000),
        ("Q-learning agent", 100_000),
        ("Q-learning agent", 500_000),
        ("Q-learning agent", 1_000_000),
    ]

    for policy, training_budget in row_order:
        row = find_row(summary_rows, policy, training_budget)

        # Shorten "Q-learning agent" to "Q-learning" for the table.
        table_policy_name = policy.replace(" agent", "")

        table_rows.append({
            "Policy": table_policy_name,
            "Training episodes": format_training_budget(training_budget),
            "Mean reward": f"{float(row['mean_reward']):.4f}",
            "Std. dev.": f"{float(row['standard_deviation']):.4f}",
            "Win rate": f"{get_rate_percent(row, 'win_rate'):.2f}%",
            "Loss rate": f"{get_rate_percent(row, 'loss_rate'):.2f}%",
            "Draw rate": f"{get_rate_percent(row, 'draw_rate'):.2f}%",
        })

    return table_rows


def save_table(table_rows: list[dict], filename: str):
    """Save the formatted table as a CSV file."""

    fieldnames = [
        "Policy",
        "Training episodes",
        "Mean reward",
        "Std. dev.",
        "Win rate",
        "Loss rate",
        "Draw rate",
    ]

    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(table_rows)


if __name__ == "__main__":
    # Read the summary results.
    summary_rows = read_summary(INPUT_FILE)

    # Create the cleaned and rounded table.
    table_rows = create_table_rows(summary_rows)

    # Save the table to a new CSV file.
    save_table(table_rows, OUTPUT_FILE)

    print("Results table saved to", OUTPUT_FILE)