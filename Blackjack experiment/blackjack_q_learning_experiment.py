from collections import defaultdict
import statistics
import csv
import gymnasium as gym
import numpy as np
from tqdm import tqdm


TRAINING_BUDGETS = [10_000, 50_000, 100_000, 500_000, 1_000_000]
SEEDS = range(10)
EVALUATION_EPISODES = 100_000

# Hyperparameters for the Q-learning agent
LEARNING_RATE = 0.01
START_EPSILON = 1.0
FINAL_EPSILON = 0.1
DISCOUNT_FACTOR = 1.0


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 1.0,
    ):
        """Create a Q-learning agent for the Blackjack environment."""
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(2))  # Creates the Q-table

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Select an action using an epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return int(np.argmax(self.q_values[obs]))  # Exploit

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update the Q-values based on the observed transition."""
        # If the episode ended, there is no future Q-value.
        if terminated:
            future_q_value = 0.0
        else: future_q_value = np.max(self.q_values[next_obs])

        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce epsilon after each training episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


STICK = 0
HIT = 1


def random_policy(obs, env):
    """Select a random action, ignoring the current observation."""
    return env.action_space.sample()


hard_standing_numbers = {
    1: 17,
    2: 13,
    3: 13,
    4: 12,
    5: 12,
    6: 12,
    7: 17,
    8: 17,
    9: 17,
    10: 17,
}

soft_standing_numbers = {
    1: 18,
    2: 18,
    3: 18,
    4: 18,
    5: 18,
    6: 18,
    7: 18,
    8: 18,
    9: 19,
    10: 19,
}


def reduced_basic_strategy(obs, env):
    """Select an action using a simplified Blackjack basic strategy."""
    player_sum, dealer_card, usable_ace = obs

    if usable_ace:
        minimum_standing_number = soft_standing_numbers[dealer_card]
    else:
        minimum_standing_number = hard_standing_numbers[dealer_card]

    if player_sum < minimum_standing_number:
        return HIT
    else:
        return STICK


def evaluate_policy(policy_function, number_of_episodes: int, seed: int) -> dict:
    """Evaluate a fixed policy over a given number of episodes."""
    env = gym.make("Blackjack-v1", natural=True, sab=False)
    env.action_space.seed(seed)

    episode_rewards = []
    wins = 0
    losses = 0
    draws = 0

    # tqdm displays a progress bar while the evaluation episodes run.
    for episode in tqdm(
        range(number_of_episodes),
        desc=f"Evaluating seed {seed}",
        leave=False,
    ):
        # Use a different environment seed for each evaluation episode.
        episode_seed = seed * number_of_episodes + episode
        obs, info = env.reset(seed=episode_seed)
        done = False
        episode_reward = 0

        while not done:
            # Select an action according to the policy being evaluated.
            action = policy_function(obs, env)

            obs, reward, terminated, truncated, info = env.step(action)
            reward = float(reward)

            episode_reward = episode_reward + reward
            done = terminated or truncated

        # Store the total reward from this episode.
        episode_rewards.append(episode_reward)

        # Count whether the episode was won, lost or drawn.
        if episode_reward > 0:
            wins = wins + 1
        elif episode_reward < 0:
            losses = losses + 1
        else:
            draws = draws + 1


    env.close()

    average_reward = sum(episode_rewards) / number_of_episodes

    return {
        "average_reward": average_reward,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }



def train_q_learning_agent(n_episodes: int, seed: int) -> BlackjackAgent:
    """Train one Q-learning agent for a given number of episodes."""
    env = gym.make("Blackjack-v1", natural=True, sab=False)

    # Seed NumPy and the action space to make exploration reproducible.
    np.random.seed(seed)
    env.action_space.seed(seed)

    # Epsilon is reduced linearly during the first half of training.
    epsilon_decay = (START_EPSILON - FINAL_EPSILON) / (n_episodes / 2)

    agent = BlackjackAgent(
        env=env,
        learning_rate=LEARNING_RATE,
        initial_epsilon=START_EPSILON,
        epsilon_decay=epsilon_decay,
        final_epsilon=FINAL_EPSILON,
        discount_factor=DISCOUNT_FACTOR,
    )

    # tqdm displays a progress bar while the training episodes run.
    for episode in tqdm(
        range(n_episodes),
        desc=f"Training seed {seed}, budget {n_episodes}",
        leave=False,
    ):
        # Use a different environment seed for each training episode.
        episode_seed = seed * n_episodes + episode
        obs, info = env.reset(seed=episode_seed)
        done = False

        while not done:
            # Choose an action using the current epsilon-greedy policy.
            action = agent.get_action(obs)

            # Apply the action in the environment.
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward = float(reward)

            # Update the Q-table using the observed transition.
            agent.update(
                obs=obs,
                action=action,
                reward=reward,
                terminated=terminated,
                next_obs=next_obs,
            )

            done = terminated or truncated
            obs = next_obs

        # Reduce exploration after each episode.
        agent.decay_epsilon()

    env.close()
    return agent


def evaluate_trained_agent(
    agent: BlackjackAgent,
    number_of_episodes: int,
    seed: int,
) -> dict:
    """Evaluate a trained Q-learning agent using its greedy policy."""
    env = gym.make("Blackjack-v1", natural=True, sab=False)

    episode_rewards = []
    wins = 0
    losses = 0
    draws = 0

    # tqdm displays a progress bar while the evaluation episodes run.
    for episode in tqdm(
        range(number_of_episodes),
        desc=f"Evaluating trained agent seed {seed}",
        leave=False,
    ):
        # Use a different environment seed for each evaluation episode.
        episode_seed = seed * number_of_episodes + episode
        obs, info = env.reset(seed=episode_seed)
        done = False
        episode_reward = 0

        while not done:
            # Agent selects the action with the highest Q-value.
            action = int(np.argmax(agent.q_values[obs]))

            obs, reward, terminated, truncated, info = env.step(action)
            reward = float(reward)

            episode_reward = episode_reward + reward
            done = terminated or truncated

        # Store the total reward from this episode.
        episode_rewards.append(episode_reward)

        # Count whether the episode was won, lost or drawn.
        if episode_reward > 0:
            wins = wins + 1
        elif episode_reward < 0:
            losses = losses + 1
        else:
            draws = draws + 1

    env.close()

    average_reward = sum(episode_rewards) / number_of_episodes
    return {
        "average_reward": average_reward,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }


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


def save_results_to_csv(results, filename: str):
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
    # Run this part only when the file is executed directly.
    results = []

    print("Starting baseline experiments")
    print("---")
    run_baseline_experiments(results)

    print("Starting Q-learning experiments")
    print("---")
    run_q_learning_experiments(results)

    save_results_to_csv(results, "final_experiment_results.csv")
    print("Results saved to final_experiment_results.csv")