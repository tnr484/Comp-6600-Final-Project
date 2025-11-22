import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy import stats
import os

# --- Constants ---
ACTIONS = ["R", "P", "S"]
NUM_ACTIONS = 3


# --- Utility functions ---
def outcome(a, b):
    """Return reward for a vs b: +1 win, 0 tie, -1 loss"""
    return 0 if a == b else (1 if (a - b) % 3 == 1 else -1)


# --- Opponents ---
class RandomOpponent:
    def __init__(self, probs=None):
        self.probs = np.array(probs) / np.sum(probs) if probs is not None else np.ones(3) / 3

    def act(self, history):
        return np.random.choice(3, p=self.probs)


class CyclicOpponent:
    def __init__(self, start=0):
        self.next_action = start

    def act(self, history):
        a = self.next_action
        self.next_action = (self.next_action + 1) % 3
        return a


class DeterministicPatternOpponent:
    def __init__(self, pattern):
        self.pattern = list(pattern)
        self.i = 0

    def act(self, history):
        a = self.pattern[self.i % len(self.pattern)]
        self.i += 1
        return a


# --- Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.0, epsilon=0.2, epsilon_decay=0.9995, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self.last_state = None
        self.last_action = None

    def state_from_history(self, history):
        return -1 if not history else history[-1][1]

    def act(self, history):
        s = self.state_from_history(history)
        if np.random.rand() < self.epsilon:
            a = np.random.randint(NUM_ACTIONS)
        else:
            qvals = self.Q[s]
            a = int(np.argmax(qvals + np.random.rand(NUM_ACTIONS) * 1e-8))  # tie-breaker noise
        self.last_state, self.last_action = s, a
        return a

    def learn(self, reward, new_history):
        new_s = self.state_from_history(new_history)
        old_q = self.Q[self.last_state][self.last_action]
        best_next = np.max(self.Q[new_s]) if self.gamma != 0 else 0
        self.Q[self.last_state][self.last_action] += self.alpha * (reward + self.gamma * best_next - old_q)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# --- Experiment runner ---
def run_episode(agent, opponent, rounds=1000):
    history, rewards = [], []
    for _ in range(rounds):
        a, o = agent.act(history), opponent.act(history)
        r = outcome(a, o)
        rewards.append(r)
        history.append((a, o))
        agent.learn(r, history)
    return rewards


def evaluate(agent_factory, opponent_factory, episodes=30, rounds=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    win_rates = []
    for _ in range(episodes):
        agent = agent_factory()
        opponent = opponent_factory()
        rewards = run_episode(agent, opponent, rounds)
        win_rates.append(np.mean(np.array(rewards) == 1))
    return np.array(win_rates)


def run_experiment(agent_factory, opponents, episodes=20, rounds=2000, seed=42):
    results = {}
    for name, opp_factory in opponents.items():
        print(f"Running vs {name}")
        win_rates = evaluate(agent_factory, opp_factory, episodes, rounds, seed)
        mean, std = win_rates.mean(), win_rates.std(ddof=1)
        results[name] = {
            "win_rates": win_rates,
            "mean": mean,
            "std": std
        }
        print(f"  mean win rate: {mean:.3f}, std: {std:.3f}")
    return results


# --- Visualization ---
def save_visualizations(results, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    # Histogram of win rates
    plt.figure(figsize=(8, 5))
    for name, res in results.items():
        plt.hist(res["win_rates"], bins=10, alpha=0.5, label=f"{name} (μ={res['mean']:.3f})")
    plt.xlabel("Win rate per episode")
    plt.ylabel("Count")
    plt.title("Agent win rate distribution vs opponents")
    plt.legend()
    hist_path = os.path.join(output_dir, "winrate_histogram.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Saved histogram to {hist_path}")

    # Boxplot for qualitative visualization
    plt.figure(figsize=(8, 5))
    plt.boxplot(
        [res["win_rates"] for res in results.values()],
        tick_labels=list(results.keys()),
        showmeans=True
    )
    plt.ylabel("Win rate per episode")
    plt.title("Win rate distributions (boxplot)")
    box_path = os.path.join(output_dir, "winrate_boxplot.png")
    plt.savefig(box_path)
    plt.close()
    print(f"Saved boxplot to {box_path}")


# --- Main ---
if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    agent_factory = lambda: QLearningAgent(alpha=0.1, gamma=0.0, epsilon=0.2, epsilon_decay=0.999, min_epsilon=0.01)

    opponents = {
        "uniform_random": lambda: RandomOpponent(),
        "biased_rock_60": lambda: RandomOpponent([0.6, 0.2, 0.2]),
        "cyclic": lambda: CyclicOpponent(),
        "pattern_RRPPSS": lambda: DeterministicPatternOpponent([0, 0, 1, 1, 2, 2])
    }

    results = run_experiment(agent_factory, opponents)

    # Save visualizations for LaTeX report
    save_visualizations(results)

    # Paired t-tests vs baseline
    baseline_wr = results["uniform_random"]["win_rates"]
    for name, res in results.items():
        if name == "uniform_random":
            continue
        tstat, pval = stats.ttest_rel(res["win_rates"], baseline_wr)
        print(f"Paired t-test {name} vs uniform_random: t={tstat:.3f}, p={pval:.4f}")
