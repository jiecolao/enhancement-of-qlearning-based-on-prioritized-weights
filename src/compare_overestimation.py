"""SOP 3: reproducible QLBPW / EQLBPW overestimation comparison.

Run from the repository root: python -m src.compare_overestimation
No existing simulator, tracker, or agent is modified. See docs/overestimation-demo.md.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from .EQLBPW.agent import Agent as EQLBPWAgent, ReplayBuffer as ProductionReplayBuffer
from .QLBPW.agent import Agent as QLBPWAgent
from .env_settings import PRESET_ENVIRONMENTS

ROOT = Path(__file__).resolve().parents[1]
GAMMA = 0.95
ALGORITHMS = ("QLBPW", "EQLBPW")
EQLBPW_SETTINGS = dict(
    state_dim=29, action_dim=4, learning_rate=0.0005, gamma=GAMMA,
    priority_alpha=0.6, beta_start=0.4, beta_end=1.0,
    e=1.0, e_min=0.05, e_decay=0.995, max_buffer=50000,
    batch_size=64, target_sync_freq=20,
    collision_weight=1.0, goal_weight=2.0, distance_weight=0.5,
)
QLBPW_SETTINGS = dict(
    alpha=0.1, gamma=GAMMA, beta=0.3, e=0.9, no_of_actions=4,
    max_buffer=20, batch_size=2000,
)


class IndexedReplayBuffer(ProductionReplayBuffer):
    """Cache chronological priorities and O(1) transition access for this demo.

    The original deque remains the source of transition dictionaries. Identical
    probabilities, RNG calls, sampled order, weights and priority updates are
    retained, including the production sampler's without-replacement behavior.
    This adapter is only for fresh training, not loading checkpoints.
    """

    def __init__(self, max_buffer, batch_size):
        super().__init__(max_buffer, batch_size)
        self._capacity = max_buffer
        self._entries = [None] * max_buffer
        self._priorities = np.empty(max_buffer, dtype=np.float64)
        self._head = 0

    def push(self, *args, **kwargs):
        super().push(*args, **kwargs)
        self._entries[self._head] = self.buffer[-1]
        self._priorities[self._head] = self.max_priority
        self._head = (self._head + 1) % self._capacity

    def sample(self, priority_alpha):
        size = len(self)
        start = self._head if size == self._capacity else 0
        priorities = (np.concatenate((self._priorities[start:], self._priorities[:start]))
                      if start else self._priorities[:size])
        scaled = priorities ** priority_alpha
        probabilities = scaled / scaled.sum()
        indices = np.random.choice(size, size=self.batch_size, replace=False, p=probabilities)
        batch = [self._entries[(start + int(i)) % self._capacity] for i in indices]
        return batch, indices, probabilities[indices]

    def update_priorities(self, indices, priorities):
        start = self._head if len(self) == self._capacity else 0
        for index, priority in zip(indices, priorities):
            slot = (start + int(index)) % self._capacity
            priority = float(priority)
            self._entries[slot]["priority"] = priority
            self._priorities[slot] = priority
            self.max_priority = max(self.max_priority, priority)


class Task:
    """Exact expected transition model, separate from reward-noise generation."""

    def __init__(self, name, preset=None):
        self.name = name
        self.diagnostic = name == "diagnostic"
        preset = PRESET_ENVIRONMENTS[0] if preset is None else preset
        self.grid = int(preset["grid_size"])
        self.start = (0, 0) if self.diagnostic else tuple(preset["start_state"])
        self.goal = (-1, -1) if self.diagnostic else tuple(preset["end_state"])
        self.obstacles = set() if self.diagnostic else set(preset["obstacles"])
        self.noise_std = 1.0 if self.diagnostic else 0.5
        self.cap = 10 if self.diagnostic else 162
        self.states = self._reachable_states()
        self.indices = {state: i for i, state in enumerate(self.states)}
        self.features = {state: self.encode(state) for state in self.states}
        self.features[self.goal] = self.encode(self.goal)
        self.reference, self.residual = self.solve()

    def transition(self, state, action):
        if self.diagnostic:
            return state, 0.0, False, False, 0.0
        if state == self.goal:
            return state, 0.0, True, False, 0.0
        dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[action]
        attempted = (min(max(state[0] + dx, 0), self.grid - 1),
                     min(max(state[1] + dy, 0), self.grid - 1))
        collision = attempted in self.obstacles
        nxt = state if collision else attempted
        done = nxt == self.goal
        reward = -1.0 if collision else (1.0 if done else 0.0)
        progress = sum(abs(a - b) for a, b in zip(state, self.goal)) - sum(
            abs(a - b) for a, b in zip(nxt, self.goal))
        return nxt, reward, done, collision, float(progress)

    def encode(self, state):
        if self.diagnostic:
            # One constant state, same four-action 29-input production network.
            return np.zeros(29, dtype=np.float32)
        x, y = state
        features = [x / (self.grid - 1), y / (self.grid - 1),
                    self.goal[0] / (self.grid - 1), self.goal[1] / (self.grid - 1)]
        features.extend(float(not (0 <= x + dx < self.grid and 0 <= y + dy < self.grid)
                              or (x + dx, y + dy) in self.obstacles)
                        for dx in range(-2, 3) for dy in range(-2, 3))
        return np.asarray(features, dtype=np.float32)

    def _reachable_states(self):
        if self.diagnostic:
            return [self.start]
        if self.start in self.obstacles or self.goal in self.obstacles:
            raise ValueError("Start and goal must be free cells")
        seen, queue = {self.start}, deque([self.start])
        while queue:
            state = queue.popleft()
            if state == self.goal:
                continue
            for action in range(4):
                nxt = self.transition(state, action)[0]
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        if self.goal not in seen:
            raise ValueError("The benchmark goal is unreachable")
        return sorted(seen - {self.goal})

    def solve(self):
        q = np.zeros((len(self.states), 4), dtype=np.float64)
        if self.diagnostic:
            return q, 0.0
        rewards, next_indices, masks = [], [], []
        for state in self.states:
            transitions = [self.transition(state, a) for a in range(4)]
            rewards.append([t[1] for t in transitions])
            next_indices.append([self.indices.get(t[0], 0) for t in transitions])
            masks.append([not t[2] for t in transitions])
        rewards, next_indices, masks = map(np.asarray, (rewards, next_indices, masks))
        for _ in range(10000):
            updated = rewards + GAMMA * masks * q.max(axis=1)[next_indices]
            residual = float(np.abs(updated - q).max())
            if residual < 1e-10:
                return q, residual
            q = updated
        raise RuntimeError("Value iteration did not converge")


def make_agent(algorithm, reference_replay=False):
    if algorithm == "QLBPW":
        return QLBPWAgent(**QLBPW_SETTINGS)
    agent = EQLBPWAgent(**EQLBPW_SETTINGS)
    if not reference_replay:
        agent.memory = IndexedReplayBuffer(agent.max_buffer, agent.batch_size)
    return agent


def q_values(agent, task, algorithm):
    if algorithm == "QLBPW":
        return np.asarray([agent.Q.get(s, np.zeros(4)) for s in task.states])
    with torch.no_grad():
        return agent.main_net(torch.from_numpy(np.stack(
            [task.features[s] for s in task.states]))).numpy().astype(np.float64)


def greedy_path(q, task):
    state, path, collisions = task.start, [task.start], 0
    for _ in range(task.cap):
        action = int(q[task.indices[state]].argmax())
        state, _, done, collision, _ = task.transition(state, action)
        collisions += int(collision)
        path.append(state)
        if done:
            return True, len(path) - 1, collisions, path
    return False, len(path) - 1, collisions, path


def evaluate(agent, task, algorithm, seed, step):
    q = q_values(agent, task, algorithm)
    if not np.isfinite(q).all():
        raise FloatingPointError(f"Nonfinite Q-values: {task.name}, {algorithm}, seed {seed}")
    selected = q.argmax(axis=1)
    index = np.arange(len(task.states))
    errors = q[index, selected] - task.reference[index, selected]
    row = dict(task=task.name, algorithm=algorithm, seed=seed, interactions=step,
               mean_max_q=float(q.max(axis=1).mean()),
               signed_bias=float(errors.mean()),
               positive_bias=float(np.maximum(errors, 0).mean()),
               mean_absolute_q_error=float(np.abs(q - task.reference).mean()),
               goal_success=None, path_steps=None, collisions=None)
    path = None
    if not task.diagnostic:
        success, steps, collisions, path = greedy_path(q, task)
        row.update(goal_success=int(success), path_steps=steps, collisions=collisions)
    return row, path


def train_run(task_name, algorithm, seed, steps, reference_replay=False):
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    task = Task(task_name)
    agent = make_agent(algorithm, reference_replay)
    # Dedicated environment RNG: every interaction has the same paired noise
    # draw regardless of replay draws, policy actions, or terminal events.
    noise = np.random.default_rng(np.random.SeedSequence([seed, 731])).normal(
        0.0, task.noise_std, size=steps)
    interval = 10 if task.diagnostic else 1620
    rows = [evaluate(agent, task, algorithm, seed, 0)[0]]
    state, episode_steps, episodes, updates = task.start, 0, 0, 0
    started = last_report = time.perf_counter()
    for step in range(1, steps + 1):
        action = (agent.e_greedy(task.features[state]) if algorithm == "EQLBPW"
                  else int(agent.epsilon_greedy(state)))
        nxt, expected_reward, done, collision, progress = task.transition(state, action)
        reward = expected_reward + (0.0 if done else float(noise[step - 1]))
        if algorithm == "EQLBPW":
            agent.update_beta((step - 1) / max(steps - 1, 1))
            agent.memory.push(task.features[state], action, reward, task.features[nxt],
                              done, collision, done, progress)
            agent.update()
            updates += int(len(agent.memory) >= agent.batch_size)
        else:
            current = agent.Q.get(state, np.zeros(4))[action]
            target = reward + (0.0 if done else GAMMA * agent.Q.get(nxt, np.zeros(4)).max())
            agent.memory.push(state, action, reward, nxt, target - current)
            agent.update_Q(*agent.adjust_lr(), end_state=task.goal, obstacles=task.obstacles)
            updates += 1
        state = nxt
        episode_steps += 1
        if done or episode_steps == task.cap:
            episodes += 1
            if algorithm == "EQLBPW":
                agent.decay_e()
                if episodes % agent.target_sync_freq == 0:
                    agent.sync_target()
            # Production command-line QLBPW keeps epsilon fixed at 0.9.
            state, episode_steps = task.start, 0
        if step % interval == 0 or step == steps:
            rows.append(evaluate(agent, task, algorithm, seed, step)[0])
        if time.perf_counter() - last_report >= 45:
            print(f"{task_name} {algorithm} seed={seed}: {step:,}/{steps:,} interactions", flush=True)
            last_report = time.perf_counter()
    _, path = evaluate(agent, task, algorithm, seed, steps)
    return dict(task=task_name, algorithm=algorithm, seed=seed, rows=rows,
                interactions=steps, updates=updates, completed_episodes=episodes,
                elapsed_seconds=time.perf_counter() - started, final_path=path,
                final_epsilon=agent.e)


def bootstrap(values, seed=908, draws=5000):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return dict(mean=mean, ci95=None)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return dict(mean=mean, ci95=np.quantile(means, [0.025, 0.975]).tolist())


def summarize(runs):
    summary = {}
    for task in ("diagnostic", "grid"):
        task_summary = {}
        for metric in ("positive_bias", "signed_bias", "mean_absolute_q_error"):
            by_algorithm = {}
            for algorithm in ALGORITHMS:
                values = {}
                for run in runs:
                    if run["task"] == task and run["algorithm"] == algorithm:
                        checkpoints = [r for r in run["rows"] if r["interactions"] > 0]
                        window = checkpoints[-max(1, math.ceil(len(checkpoints) * 0.2)):]
                        values[run["seed"]] = float(np.mean([r[metric] for r in window]))
                by_algorithm[algorithm] = values
            seeds = sorted(by_algorithm["QLBPW"])
            paired = bootstrap([by_algorithm["QLBPW"][s] - by_algorithm["EQLBPW"][s] for s in seeds])
            task_summary[metric] = {
                a: bootstrap([by_algorithm[a][s] for s in seeds]) for a in ALGORITHMS}
            task_summary[metric]["QLBPW_minus_EQLBPW"] = paired
        interval = task_summary["positive_bias"]["QLBPW_minus_EQLBPW"]["ci95"]
        task_summary["bias_conclusion"] = (
            "Insufficient seeds for an uncertainty interval" if interval is None else
            "EQLBPW has lower positive overestimation" if interval[0] > 0 else
            "QLBPW has lower positive overestimation" if interval[1] < 0 else
            "Inconclusive: paired 95% interval includes zero")
        if task == "grid":
            task_summary["final_greedy_success"] = {
                a: bootstrap([r["rows"][-1]["goal_success"] for r in runs
                              if r["task"] == task and r["algorithm"] == a]) for a in ALGORITHMS}
        summary[task] = task_summary
    return summary


def plot_results(runs, summary, output, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def decorate(ax, title, ylabel):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

    specifications = [
        ("diagnostic", "mean_max_q", "Estimated Maximum Q-Value", "Diagnostic: QLBPW vs EQLBPW", "diagnostic_q_values.png"),
        ("grid", "positive_bias", "Mean Positive Overestimation (lower is better)", "Grid: QLBPW vs EQLBPW Overestimation", "grid_overestimation.png"),
    ]
    def curves(ax, task, metric):
        for algorithm, color in zip(ALGORITHMS, ("tab:blue", "tab:orange")):
            selected = sorted([r for r in runs if r["task"] == task and r["algorithm"] == algorithm],
                              key=lambda r: r["seed"])
            x = [r["interactions"] for r in selected[0]["rows"]]
            values = np.array([[r[metric] for r in run["rows"]] for run in selected])
            ax.plot(x, values.mean(axis=0), color=color, linewidth=2, marker="o",
                    markevery=max(1, len(x) // 10), markersize=4, label=algorithm)
            if len(selected) > 1:
                indices = np.random.default_rng(908).integers(0, len(selected), (2000, len(selected)))
                means = values[indices].mean(axis=1)
                lo, hi = np.quantile(means, [0.025, 0.975], axis=0)
                ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.set_xlabel("Training Interactions")
        ax.legend()

    for task, metric, ylabel, title, filename in specifications:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        curves(ax, task, metric)
        ax.axhline(0, color="black", linestyle="--", linewidth=1,
                   label="True optimal Q = 0" if task == "diagnostic" else "Zero overestimation")
        ax.legend()
        decorate(ax, title, ylabel)
        fig.text(0.5, 0.01, "Mean across seeds; shading: pointwise 95% bootstrap CI", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.savefig(output / filename, dpi=180)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, task in zip(axes, ("diagnostic", "grid")):
        curves(ax, task, "mean_absolute_q_error")
        decorate(ax, f"{task.title()}: Q-Value Accuracy", "Mean Absolute Q-Error (lower is better)")
    fig.tight_layout()
    fig.savefig(output / "q_value_accuracy.png", dpi=180)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, task in zip(axes[:2], ("diagnostic", "grid")):
        for x, (algorithm, color) in enumerate(zip(ALGORITHMS, ("tab:blue", "tab:orange"))):
            stat = summary[task]["positive_bias"][algorithm]
            ax.bar(x, stat["mean"], color=color, width=0.55)
            if stat["ci95"] is not None:
                low, high = stat["ci95"]
                ax.vlines(x, low, high, color="black", linewidth=1.5)
                ax.hlines([low, high], x - .06, x + .06, color="black")
        ax.set_xticks([0, 1], ALGORITHMS)
        decorate(ax, f"{task.title()}: Final-Window Bias", "Positive Overestimation")
    stats = summary["grid"]["final_greedy_success"]
    axes[2].bar(ALGORITHMS, [100 * stats[a]["mean"] for a in ALGORITHMS],
                color=["tab:blue", "tab:orange"], width=.55)
    axes[2].set_ylim(0, 105)
    decorate(axes[2], "Grid: Final Greedy Success", "Seeds Reaching Goal (%)")
    fig.text(.5, .01, "Bias: final 20% of checkpoints; error bars: 95% CI across seeds. Success: fixed start and map.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / "final_comparison.png", dpi=180)
    if show:
        plt.show()
    plt.close("all")


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--diagnostic-steps", type=int, default=5000)
    parser.add_argument("--grid-steps", type=int, default=81000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "compare_figures" / "overestimation")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--reference-replay", action="store_true",
                        help="Use the original slower deque sampler instead of the equivalent indexed adapter")
    args = parser.parse_args(argv)
    if min(args.diagnostic_steps, args.grid_steps, args.workers) < 1:
        parser.error("Step budgets and workers must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("Seeds must be unique integers in [0, 2**32)")
    output = args.output_dir.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    grid = Task("grid")
    config = dict(seeds=args.seeds, diagnostic_steps=args.diagnostic_steps, grid_steps=args.grid_steps,
                  workers=args.workers, gamma=GAMMA, QLBPW=QLBPW_SETTINGS, EQLBPW=EQLBPW_SETTINGS,
                  replay_access="production" if args.reference_replay else "equivalent indexed adapter",
                  versions=dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__),
                  grid=dict(size=grid.grid, start=grid.start, goal=grid.goal, obstacles=sorted(grid.obstacles),
                            reachable_nonterminal_states=len(grid.states), cap=grid.cap,
                            noise_std=grid.noise_std, bellman_residual=grid.residual),
                  diagnostic=dict(noise_std=1.0, block_size=10, true_q=0.0, actions=4, observation="29 zeros"),
                  interpretation="Two complete algorithms; no causal isolation of Double DQN. All seeds retained.",
                  qlbpw_epsilon="Fixed 0.9, as in current standalone simulator",
                  eqlbpw_schedule="Epsilon decay and target sync per episode / diagnostic block; beta per interaction",
                  ci_method="Percentile bootstrap across independent seeds; paired differences; final 20% checkpoints",
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), ROOT / "src/EQLBPW/agent.py",
                                           ROOT / "src/QLBPW/agent.py", ROOT / "src/env_settings.py"]})
    import matplotlib
    config["versions"]["matplotlib"] = matplotlib.__version__
    write_json(output / "config.json", config)
    write_json(output / "reference_q.json", [dict(state=s, q=grid.reference[i].tolist()) for i, s in enumerate(grid.states)])
    print(f"Results: {output}\nRunning {len(args.seeds)} seeds; {args.workers} worker(s).", flush=True)
    runs = []
    jobs = [(task, a, seed, steps, args.reference_replay) for task, steps in (("diagnostic", args.diagnostic_steps), ("grid", args.grid_steps))
            for seed in args.seeds for a in ALGORITHMS]
    def save_run(run):
        runs.append(run)
        write_json(output / f"{run['task']}_{run['algorithm']}_seed{run['seed']}.json", run)
        print(f"Completed {run['task']} {run['algorithm']} seed={run['seed']} "
              f"({run['elapsed_seconds']:.1f}s); {len(runs)}/{len(jobs)} runs", flush=True)
    if args.workers == 1:
        for job in jobs:
            save_run(train_run(*job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(train_run, *job) for job in jobs]
            for future in as_completed(futures):
                save_run(future.result())
    runs.sort(key=lambda r: (r["task"], r["seed"], r["algorithm"]))
    with (output / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]["rows"][0]))
        writer.writeheader()
        for run in runs:
            writer.writerows(run["rows"])
    summary = summarize(runs)
    write_json(output / "summary.json", summary)
    plot_results(runs, summary, output, not args.no_show)
    for task in ("diagnostic", "grid"):
        print(f"{task}: {summary[task]['bias_conclusion']}")
    print(f"Saved figures, measurements and settings to {output}")
    return output


if __name__ == "__main__":
    main()
