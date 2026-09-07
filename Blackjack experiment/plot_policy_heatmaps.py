from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from blackjack_q_learning_experiment import (
    reduced_basic_strategy,
    train_q_learning_agent,
)


# The heatmap figure will be saved in this folder.
OUTPUT_FOLDER = Path("figures")

# Use the 1,000,000 episode seed whose evaluation is closest to the mean.
TRAINING_BUDGET = 1_000_000
SEED = 2

# The Blackjack state space that will be shown in the heatmaps.
PLAYER_SUMS = list(range(12, 22))
PLAYER_SUM_LABELS = [str(player_sum) for player_sum in PLAYER_SUMS]

DEALER_CARDS = list(range(1, 11))
DEALER_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


def get_q_learning_policy_matrix(agent, usable_ace: bool) -> np.ndarray:
    """Create a policy matrix for the trained Q-learning agent."""

    policy_matrix = np.zeros((len(DEALER_CARDS), len(PLAYER_SUMS)))

    for dealer_index, dealer_card in enumerate(DEALER_CARDS):
        for player_index, player_sum in enumerate(PLAYER_SUMS):
            state = (player_sum, dealer_card, usable_ace)

            # Choose the greedy action from the learned Q-table.
            action = int(np.argmax(agent.q_values[state]))

            policy_matrix[dealer_index, player_index] = action

    return policy_matrix


def get_basic_strategy_policy_matrix(usable_ace: bool) -> np.ndarray:
    """Create a policy matrix for the reduced basic strategy."""

    policy_matrix = np.zeros((len(DEALER_CARDS), len(PLAYER_SUMS)))

    for dealer_index, dealer_card in enumerate(DEALER_CARDS):
        for player_index, player_sum in enumerate(PLAYER_SUMS):
            state = (player_sum, dealer_card, usable_ace)

            # The reduced basic strategy function ignores env, so None is fine.
            action = reduced_basic_strategy(state, None)

            policy_matrix[dealer_index, player_index] = action

    return policy_matrix


def draw_policy_heatmap(ax, policy_matrix: np.ndarray, title: str):
    """Draw one policy heatmap on the given subplot axis."""

    # 0 = Stick, 1 = Hit
    cmap = sns.color_palette(["#7f7f7f", "#8fd17f"], as_cmap=True)

    sns.heatmap(
        policy_matrix,
        ax=ax,
        cmap=cmap,
        cbar=False,
        annot=False,
        linewidths=0.5,
        linecolor="white",
        xticklabels=PLAYER_SUM_LABELS,
        yticklabels=DEALER_LABELS,
    )

    ax.set_title(title)
    ax.set_xlabel("Player sum")
    ax.set_ylabel("Dealer showing")


def plot_policy_heatmaps():
    """Train one Q-learning agent and compare its learned policy to the reduced basic strategy."""

    print("Training representative Q-learning agent...")
    agent = train_q_learning_agent(
        n_episodes=TRAINING_BUDGET,
        seed=SEED,
    )

    print("Creating policy matrices...")
    q_no_ace = get_q_learning_policy_matrix(agent, usable_ace=False)
    q_with_ace = get_q_learning_policy_matrix(agent, usable_ace=True)

    basic_no_ace = get_basic_strategy_policy_matrix(usable_ace=False)
    basic_with_ace = get_basic_strategy_policy_matrix(usable_ace=True)

    # Calculate how often the two policies choose the same action.
    agreement_no_ace = np.mean(q_no_ace == basic_no_ace)
    agreement_with_ace = np.mean(q_with_ace == basic_with_ace)

    overall_agreement = np.mean(
        np.concatenate([
            (q_no_ace == basic_no_ace).flatten(),
            (q_with_ace == basic_with_ace).flatten(),
        ])
    )

    print(f"Agreement without usable ace: {agreement_no_ace:.1%}")
    print(f"Agreement with usable ace: {agreement_with_ace:.1%}")
    print(f"Overall agreement: {overall_agreement:.1%}")

    print("Plotting heatmaps...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    draw_policy_heatmap(
        axes[0, 0],
        q_no_ace,
        "Q-learning: without usable ace",
    )
    draw_policy_heatmap(
        axes[0, 1],
        q_with_ace,
        "Q-learning: with usable ace",
    )
    draw_policy_heatmap(
        axes[1, 0],
        basic_no_ace,
        "Reduced basic strategy: without usable ace",
    )
    draw_policy_heatmap(
        axes[1, 1],
        basic_with_ace,
        "Reduced basic strategy: with usable ace",
    )

    # Add one shared legend for all subplots.
    legend_handles = [
        Patch(facecolor="#8fd17f", edgecolor="black", label="Hit"),
        Patch(facecolor="#7f7f7f", edgecolor="black", label="Stick"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=True,
    )

    fig.suptitle(
        "Policy comparison: Q-learning agent vs. reduced basic strategy",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout(rect=(0, 0.05, 1, 0.94))

    plt.savefig(
        OUTPUT_FOLDER / "policy_heatmap_comparison.png",
        dpi=300,
    )

    plt.close()


if __name__ == "__main__":
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    plot_policy_heatmaps()
    print("Figure saved in the figures folder.")