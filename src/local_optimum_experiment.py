from collections import deque
import random

import numpy as np
import torch

from EQLBPW.simulator import simulate as EQLBPW_simulate
from QLBPW.simulator import simulate as QLBPW_simulate
from visualizer import Visualizer


# Fixed map with multiple valid routes. The horizontal obstacle wall creates
# two route choices: a direct crossing near the right side and a longer
# detour through the left-side opening.
LOCAL_OPTIMUM_PRESET = {
    "name": "Local Optimum Comparison Test",
    "grid_size": 12,
    "start_state": (0, 0),
    "end_state": (11, 11),
    "obstacles": {
        (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5),
        (1, 4), (1, 6),
    },
}


def _bfs_shortest_path(env):
    """Return one shortest valid path using BFS."""
    queue = deque([(env.start_state, [env.start_state])])
    visited = {env.start_state}

    while queue:
        state, path = queue.popleft()
        if state == env.end_state:
            return path

        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            next_state = (state[0] + dx, state[1] + dy)

            if not (
                0 <= next_state[0] < env.grid_cols
                and 0 <= next_state[1] < env.grid_rows
            ):
                continue

            if next_state in env.obstacles or next_state in visited:
                continue

            visited.add(next_state)
            queue.append((next_state, path + [next_state]))

    return None


def _greedy_path(agent, env, is_dqn):
    """Trace the learned greedy policy without exploration."""
    state = env.start_state
    path = [state]
    total_reward = 0.0
    max_steps = env.grid_rows * env.grid_cols * 3

    for _ in range(max_steps):
        if is_dqn:
            # EQLBPW's network expects its 29-dimensional environment state,
            # not the tuple position used by QLBPW.
            env.agent_pos = state
            state_tensor = torch.as_tensor(
                env.get_state(), dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                action = agent.main_net(state_tensor).argmax(dim=1).item()
        else:
            q_values = agent.Q.get(state, np.zeros(agent.no_of_actions))
            action = int(np.argmax(q_values))

        result = env.take_step(state, action)

        # QLBPW returns 3 values; EQLBPW additionally returns info.
        next_state, reward, is_terminal = result[:3]
        path.append(next_state)
        total_reward += reward
        state = next_state

        if is_terminal:
            break

    return path, total_reward


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_preset():
    """Return a fresh copy so experiments cannot mutate the shared preset."""
    return {
        "name": LOCAL_OPTIMUM_PRESET["name"],
        "grid_size": LOCAL_OPTIMUM_PRESET["grid_size"],
        "start_state": LOCAL_OPTIMUM_PRESET["start_state"],
        "end_state": LOCAL_OPTIMUM_PRESET["end_state"],
        "obstacles": set(LOCAL_OPTIMUM_PRESET["obstacles"]),
    }


def _summarize_runs(runs):
    successful = [run for run in runs if run["goal_reached"]]
    optimal = [run for run in successful if run["optimal"]]
    suboptimal = [run for run in successful if not run["optimal"]]

    return {
        "success_rate": len(successful) / len(runs) if runs else 0.0,
        "optimal_solution_rate": (
            len(optimal) / len(successful) if successful else 0.0
        ),
        "suboptimal_solution_rate": (
            len(suboptimal) / len(successful) if successful else 0.0
        ),
        "average_path_length": (
            np.mean([run["learned_steps"] for run in successful])
            if successful else 0.0
        ),
        "average_optimality": (
            np.mean([run["optimality"] for run in successful])
            if successful else 0.0
        ),
    }


def run_local_optimum_experiment(
        episodes=100,
        seeds=(0, 1, 2, 3, 4),
        save_fig=False,
):
    """Compare QLBPW and EQLBPW on the same fixed local-optimum map.

    BFS defines the global shortest valid path. Each algorithm is trained
    independently for the same number of episodes and random seeds, then
    evaluated greedily without exploration.
    """
    preset = _make_preset()

    # Build a lightweight environment through the QLBPW simulator so that
    # the BFS reference uses exactly the same grid and obstacle semantics.
    # The simulator is not run here; the preset itself is sufficient for BFS.
    class _GridReference:
        grid_rows = preset["grid_size"]
        grid_cols = preset["grid_size"]
        start_state = preset["start_state"]
        end_state = preset["end_state"]
        obstacles = preset["obstacles"]

    shortest_path = _bfs_shortest_path(_GridReference())
    if shortest_path is None:
        raise ValueError("The local-optimum test map has no valid path.")

    shortest_steps = len(shortest_path) - 1

    print("\n" + "=" * 70)
    print("QLBPW vs EQLBPW Local-Optimum Experiment")
    print("=" * 70)
    print(f"Grid: {preset['grid_size']} x {preset['grid_size']}")
    print(f"BFS shortest path: {shortest_steps} steps")
    print(f"Episodes per seed: {episodes}")
    print(f"Seeds: {tuple(seeds)}")
    print("\nEach algorithm is evaluated against the same BFS optimum.")

    all_runs = []
    representative_models = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # QLBPW
        _seed_everything(seed)
        qlbpw_agent, qlbpw_env = QLBPW_simulate(
            grid_size=preset["grid_size"],
            episodes=episodes,
            preset=preset,
        )
        qlbpw_path, qlbpw_reward = _greedy_path(
            qlbpw_agent, qlbpw_env, is_dqn=False
        )
        qlbpw_steps = len(qlbpw_path) - 1
        qlbpw_success = qlbpw_path[-1] == qlbpw_env.end_state
        qlbpw_optimal = qlbpw_success and qlbpw_steps == shortest_steps
        qlbpw_optimality = (
            shortest_steps / qlbpw_steps
            if qlbpw_success and qlbpw_steps > 0 else 0.0
        )
        qlbpw_run = {
            "algorithm": "QLBPW",
            "seed": seed,
            "goal_reached": qlbpw_success,
            "learned_steps": qlbpw_steps,
            "shortest_steps": shortest_steps,
            "optimal": qlbpw_optimal,
            "optimality": qlbpw_optimality,
            "total_reward": qlbpw_reward,
            "path": qlbpw_path,
        }
        all_runs.append(qlbpw_run)

        # EQLBPW
        _seed_everything(seed)
        eqlbpw_agent, eqlbpw_env = EQLBPW_simulate(
            grid_size=preset["grid_size"],
            episodes=episodes,
            preset=preset,
        )
        eqlbpw_path, eqlbpw_reward = _greedy_path(
            eqlbpw_agent, eqlbpw_env, is_dqn=True
        )
        eqlbpw_steps = len(eqlbpw_path) - 1
        eqlbpw_success = eqlbpw_path[-1] == eqlbpw_env.end_state
        eqlbpw_optimal = eqlbpw_success and eqlbpw_steps == shortest_steps
        eqlbpw_optimality = (
            shortest_steps / eqlbpw_steps
            if eqlbpw_success and eqlbpw_steps > 0 else 0.0
        )
        eqlbpw_run = {
            "algorithm": "EQLBPW",
            "seed": seed,
            "goal_reached": eqlbpw_success,
            "learned_steps": eqlbpw_steps,
            "shortest_steps": shortest_steps,
            "optimal": eqlbpw_optimal,
            "optimality": eqlbpw_optimality,
            "total_reward": eqlbpw_reward,
            "path": eqlbpw_path,
        }
        all_runs.append(eqlbpw_run)

        representative_models.setdefault(
            "QLBPW", (qlbpw_agent, qlbpw_env)
        )
        representative_models.setdefault(
            "EQLBPW", (eqlbpw_agent, eqlbpw_env)
        )

        print(
            f"QLBPW : goal={qlbpw_success}, steps={qlbpw_steps}, "
            f"optimal={qlbpw_optimal}, optimality={qlbpw_optimality:.3f}, "
            f"reward={qlbpw_reward}"
        )
        print(
            f"EQLBPW: goal={eqlbpw_success}, steps={eqlbpw_steps}, "
            f"optimal={eqlbpw_optimal}, optimality={eqlbpw_optimality:.3f}, "
            f"reward={eqlbpw_reward}"
        )

    qlbpw_runs = [run for run in all_runs if run["algorithm"] == "QLBPW"]
    eqlbpw_runs = [run for run in all_runs if run["algorithm"] == "EQLBPW"]

    summary = {
        "QLBPW": _summarize_runs(qlbpw_runs),
        "EQLBPW": _summarize_runs(eqlbpw_runs),
    }

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for algorithm in ("QLBPW", "EQLBPW"):
        metrics = summary[algorithm]
        print(
            f"{algorithm}: "
            f"success={metrics['success_rate']:.2%}, "
            f"optimal={metrics['optimal_solution_rate']:.2%}, "
            f"suboptimal={metrics['suboptimal_solution_rate']:.2%}, "
            f"avg_path={metrics['average_path_length']:.2f}, "
            f"avg_optimality={metrics['average_optimality']:.3f}"
        )

    results = {
        "preset": preset,
        "shortest_path": shortest_path,
        "shortest_steps": shortest_steps,
        "runs": all_runs,
        "summary": summary,
        "representative_models": representative_models,
    }

    if save_fig:
        plot_local_optimum_results(results, save_fig=True)
        visualize_local_optimum_paths(results, save_fig=True)

    return results


def plot_local_optimum_results(results, save_fig=False):
    """Plot optimal-solution rate and learned path length by seed."""
    visualizer = Visualizer(agent=None, env=None)
    summary = results["summary"]
    seeds = sorted({run["seed"] for run in results["runs"]})

    visualizer.plot_bar(
        categories=["QLBPW", "EQLBPW"],
        values=[
            summary["QLBPW"]["optimal_solution_rate"],
            summary["EQLBPW"]["optimal_solution_rate"],
        ],
        title="Optimal-Solution Rate on Local-Optimum Test",
        xlabel="Algorithm",
        ylabel="Optimal-Solution Rate",
        save_fig=save_fig,
    )

    series = {}
    for algorithm in ("QLBPW", "EQLBPW"):
        runs = [
            run for run in results["runs"]
            if run["algorithm"] == algorithm
        ]
        by_seed = {run["seed"]: run["learned_steps"] for run in runs}
        series[algorithm] = [by_seed[seed] for seed in seeds]

    visualizer.compare_plot_line(
        x=seeds,
        series=series,
        title="Learned Path Length by Seed",
        xlabel="Seed",
        ylabel="Path Length (steps)",
        save_fig=save_fig,
    )


def visualize_local_optimum_paths(results, save_fig=False):
    """Visualize representative greedy paths using the existing Visualizer."""
    qlbpw_agent, qlbpw_env = results["representative_models"]["QLBPW"]
    eqlbpw_agent, eqlbpw_env = results["representative_models"]["EQLBPW"]

    qlbpw_visualizer = Visualizer(qlbpw_agent, qlbpw_env)
    qlbpw_visualizer.qlbpw_visualize_learned_path(
        agent=qlbpw_agent,
        env=qlbpw_env,
        title="QLBPW Local-Optimum Test Path",
        save_fig=save_fig,
    )

    eqlbpw_visualizer = Visualizer(eqlbpw_agent, eqlbpw_env)
    eqlbpw_visualizer.eqlbpqw_visualize_learned_path(
        title="EQLBPW Local-Optimum Test Path",
        save_fig=save_fig,
    )
