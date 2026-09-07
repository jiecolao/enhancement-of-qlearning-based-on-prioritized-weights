"""Controlled stochastic overestimation benchmark.

Run from repository root with:
    python -m src.overestimation_benchmark

This runner imports the production QLBPW/EQLBPW agents and production-based
benchmark environments. It does not replace either learning algorithm.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from .EQLBPW.agent import Agent as EQLBPWAgent
from .EQLBPW.tracker import EnvironmentTracker as EQLBPWTracker
from .QLBPW.agent import Agent as QLBPWAgent
from .QLBPW.tracker import EnvironmentTracker as QLBPWTracker
from .overestimation_environment import (
    StochasticOverestimationEQLBPWEnvironment,
    StochasticOverestimationQLBPWEnvironment,
)

GAMMA = 0.95
ACTIONS = 4
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "compare_figures" / "overestimation"


def make_agents():
    q = QLBPWAgent(
        alpha=0.1, gamma=GAMMA, beta=0.3, e=0.9,
        no_of_actions=ACTIONS, max_buffer=200, batch_size=2000,
    )
    # These are the same production EQLBPW components: DQN, prioritized
    # replay, importance sampling, target network, and Double-DQN update.
    e = EQLBPWAgent(
        state_dim=29, action_dim=ACTIONS, learning_rate=0.0005,
        gamma=GAMMA, priority_alpha=0.6, beta_start=0.4, beta_end=1.0,
        e=1.0, e_min=0.05, e_decay=0.995,
        max_buffer=5000, batch_size=32, target_sync_freq=20,
        collision_weight=1.0, goal_weight=2.0, distance_weight=0.5,
    )
    # The production QLBPW tracker expects these legacy fields although the
    # current agent constructor does not initialize them.
    q.e_min = q.e
    q.e_decay = 1.0
    return q, e


def make_envs(q_agent, e_agent, reward_std, seed, episodes):
    common = dict(
        grid=2, start_state=(0, 0), end_state=(1, 1), episodes=episodes,
        ep_tracker=max(1, episodes // 10), no_of_obstacles=0,
        static_obstacles=[], is_dynamic_obs=False,
    )
    q_env = StochasticOverestimationQLBPWEnvironment(
        agent=q_agent, reward_std=reward_std,
        rng=np.random.default_rng(np.random.SeedSequence([seed, 101])), **common,
    )
    e_env = StochasticOverestimationEQLBPWEnvironment(
        agent=e_agent, reward_std=reward_std,
        rng=np.random.default_rng(np.random.SeedSequence([seed, 202])), **common,
    )
    q_env.generate_obstacles()
    e_env.generate_obstacles()
    return q_env, e_env


def expected_model(env):
    """Return the exact expected optimal Q table; noise has zero expectation."""
    states = [(x, y) for y in range(env.grid_rows) for x in range(env.grid_cols)
              if (x, y) != env.end_state]
    index = {s: i for i, s in enumerate(states)}
    q = np.zeros((len(states), ACTIONS), dtype=float)
    for _ in range(10000):
        new = np.zeros_like(q)
        for state, i in index.items():
            for action in range(ACTIONS):
                dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[action]
                attempted = (min(max(state[0] + dx, 0), env.grid_cols - 1),
                             min(max(state[1] + dy, 0), env.grid_rows - 1))
                nxt = state if attempted in env.obstacles else attempted
                done = nxt == env.end_state
                expected_reward = 1.0 if done else (-1.0 if nxt in env.obstacles else 0.0)
                new[i, action] = expected_reward + (0.0 if done else GAMMA * q[index[nxt]].max())
        if np.max(np.abs(new - q)) < 1e-12:
            return states, new
        q = new
    raise RuntimeError("Expected-value iteration did not converge")


def learned_q(agent, env, algorithm, states):
    if algorithm == "QLBPW":
        return np.asarray([agent.Q.get(s, np.zeros(ACTIONS)) for s in states], dtype=float)
    original = env.agent_pos
    features = []
    for state in states:
        env.agent_pos = state
        features.append(env.get_state())
    env.agent_pos = original
    with torch.no_grad():
        return agent.main_net(torch.as_tensor(np.asarray(features), dtype=torch.float32)).cpu().numpy()


def train_one(seed, episodes, reward_std, algorithm, checkpoints=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    q_agent, e_agent = make_agents()
    q_env, e_env = make_envs(q_agent, e_agent, reward_std, seed, episodes)
    agent, env = (q_agent, q_env) if algorithm == "QLBPW" else (e_agent, e_env)
    tracker = QLBPWTracker(agent, env) if algorithm == "QLBPW" else EQLBPWTracker(agent, env)
    states, truth = expected_model(env)
    start_i = states.index(env.start_state)
    history = []
    interval = max(1, episodes // checkpoints)

    for episode in range(1, episodes + 1):
        state = env.start_state
        env.agent_pos = state
        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < env.max_steps:
            if algorithm == "QLBPW":
                action = int(agent.epsilon_greedy(state))
                next_state, reward, done = env.take_step(state, action)
                current = agent.Q.get(state, np.zeros(ACTIONS))[action]
                next_max = 0.0 if done else agent.Q.get(next_state, np.zeros(ACTIONS)).max()
                td_error = reward + GAMMA * next_max - current
                agent.memory.push(state, action, reward, next_state, td_error)
                sample = agent.adjust_lr()
                agent.update_Q(*sample, end_state=env.end_state, obstacles=env.obstacles)
            else:
                env.agent_pos = state
                current_features = env.get_state()
                action = agent.e_greedy(current_features)
                next_state, reward, done, info = env.take_step(state, action)
                next_features = env.get_state()
                agent.memory.push(
                    current_features, action, reward, next_features, done,
                    float(info.get("collision", False)), float(info.get("goal", False)),
                    float(info.get("distance_progress", 0.0)),
                )
                agent.update()
            state = next_state
            total_reward += reward
            steps += 1

        if algorithm == "EQLBPW":
            agent.decay_e()
            if episode % agent.target_sync_freq == 0:
                agent.sync_target()

        tracker.steps_per_ep = steps
        tracker.rewards_per_ep = total_reward
        tracker.record_episode(done)

        if episode % interval == 0 or episode == episodes:
            q = learned_q(agent, env, algorithm, states)
            bias = float(q[start_i].max() - truth[start_i].max())
            history.append((episode, bias, float(q[start_i].max())))

    q = learned_q(agent, env, algorithm, states)
    start_bias = float(q[start_i].max() - truth[start_i].max())
    return {
        "seed": seed, "algorithm": algorithm, "history": history,
        "bias": start_bias, "estimated_q": float(q[start_i].max()),
        "true_q": float(truth[start_i].max()),
        "positive_overestimation": max(start_bias, 0.0),
        "success_rate": tracker.get_success_rate(),
    }


def summarize(results):
    summary = {}
    for algorithm in ("QLBPW", "EQLBPW"):
        rows = [r for r in results if r["algorithm"] == algorithm]
        bias = np.asarray([r["bias"] for r in rows], dtype=float)
        estimate = np.asarray([r["estimated_q"] for r in rows], dtype=float)
        positive = np.maximum(bias, 0.0)
        summary[algorithm] = {
            "true_q": float(np.mean([r["true_q"] for r in rows])),
            "estimated_q": float(estimate.mean()),
            "estimated_q_std": float(estimate.std(ddof=1)) if len(estimate) > 1 else 0.0,
            "mean_bias": float(bias.mean()),
            "bias_std": float(bias.std(ddof=1)) if len(bias) > 1 else 0.0,
            "positive_overestimation": float(positive.mean()),
            "positive_overestimation_std": float(positive.std(ddof=1)) if len(positive) > 1 else 0.0,
        }
    return summary


def plot_results(results, summary, save=True, show=True):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    algorithms = ["QLBPW", "EQLBPW"]
    x = np.arange(len(algorithms))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, ax = plt.subplots(figsize=(8, 5))
    means = [summary[a]["mean_bias"] for a in algorithms]
    errors = [summary[a]["bias_std"] for a in algorithms]
    ax.bar(x, means, yerr=errors, capsize=5)
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x, algorithms)
    ax.set_ylabel("Mean Q-value bias")
    ax.set_title("Overestimation Bias: QLBPW vs EQLBPW")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / f"overestimation_bias_{timestamp}.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    truth = [summary[a]["true_q"] for a in algorithms]
    estimate = [summary[a]["estimated_q"] for a in algorithms]
    ax.bar(x - width / 2, truth, width, label="True max Q")
    ax.bar(x + width / 2, estimate, width, label="Learned max Q")
    ax.set_xticks(x, algorithms)
    ax.set_ylabel("Q-value")
    ax.set_title("True vs Learned Maximum Q-value")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / f"true_vs_learned_q_{timestamp}.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_csv(results, summary):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"overestimation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "seed", "true_q", "estimated_q", "bias", "positive_overestimation", "success_rate"])
        for r in results:
            writer.writerow([r["algorithm"], r["seed"], r["true_q"], r["estimated_q"], r["bias"], r["positive_overestimation"], r["success_rate"]])
        writer.writerow([])
        writer.writerow(["algorithm", "true_q", "estimated_q", "estimated_q_std", "mean_bias", "bias_std", "positive_overestimation", "positive_overestimation_std"])
        for a in summary:
            writer.writerow([a, *summary[a].values()])
    return path


def run(seeds, episodes, reward_std, show=True):
    results = []
    for seed in seeds:
        for algorithm in ("QLBPW", "EQLBPW"):
            print(f"Running {algorithm}: seed={seed}, episodes={episodes}")
            result = train_one(seed, episodes, reward_std, algorithm)
            results.append(result)
            print(f"  true={result['true_q']:.6f} estimated={result['estimated_q']:.6f} bias={result['bias']:.6f}")

    summary = summarize(results)
    csv_path = save_csv(results, summary)
    plot_results(results, summary, show=show)
    print("\nSummary")
    for algorithm, values in summary.items():
        print(
            f"{algorithm}: estimated={values['estimated_q']:.6f} +/- {values['estimated_q_std']:.6f}, "
            f"bias={values['mean_bias']:.6f} +/- {values['bias_std']:.6f}, "
            f"positive overestimation={values['positive_overestimation']:.6f}"
        )
    print(f"Results saved to {csv_path}")
    return results, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-std", type=float, default=2.0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    run([args.seed + i for i in range(args.seeds)], args.episodes, args.reward_std, show=not args.no_show)


if __name__ == "__main__":
    main()
