from EQLBPW.agent import Agent as EQLBPWAgent
from EQLBPW.environment import Environment as EQLBPWEnvironment
from EQLBPW.simulator import simulate as EQLBPW_simulate
from QLBPW.agent import Agent as QLBPWAgent
from QLBPW.environment import Environment as QLBPWEnvironment
from QLBPW.simulator import simulate as QLBPW_simulate
from visualizer import Visualizer
from env_settings import OBSTACLES, PRESET_ENVIRONMENTS
import numpy as np
import torch
import time
import tracemalloc
import random

STATE_SPACE_PRESETS = PRESET_ENVIRONMENTS[3:9]

# Fixed 9x9 map for testing whether QLBPW can converge to a
# valid but suboptimal route when a shorter valid route exists.
# BFS should be used as the ground truth for the shortest route.
LOCAL_OPTIMUM_PRESET = {
    "name": "QLBPW Local Optimum Test",
    "grid_size": 9,
    "start_state": (0, 0),
    "end_state": (8, 8),
    "obstacles": {
        (0, 8), (1, 0), (1, 4), (2, 1), (2, 3), (2, 4),
        (3, 0), (4, 4), (4, 8), (5, 5), (6, 5), (6, 7),
        (7, 7), (8, 4), (8, 6), (8, 7),
    },
}


def _measure_simulation(simulate):
    tracemalloc.start()
    start_time = time.perf_counter()

    agent, env = simulate()

    elapsed_time = time.perf_counter() - start_time
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return agent, env, elapsed_time, peak_memory / (1024 * 1024)


def measure_elapsed_time_and_memory_usage(save_fig=False):
    qlbpw_agent, qlbpw_env, qlbpw_time, qlbpw_memory = \
        _measure_simulation(QLBPW_simulate)

    eqlbpqw_agent, eqlbpqw_env, eqlbpqw_time, eqlbpqw_memory = \
        _measure_simulation(EQLBPW_simulate)

    results = {
        "QLBPW": {
            "elapsed_time": qlbpw_time,
            "peak_memory": qlbpw_memory,
        },
        "EQLBPW": {
            "elapsed_time": eqlbpqw_time,
            "peak_memory": eqlbpqw_memory,
        },
    }

    visualizer = Visualizer(eqlbpqw_agent, eqlbpqw_env)
    visualizer.compare_plot_line(
        x=["QLBPW", "EQLBPW"],
        series={
            "Elapsed time (seconds)": [qlbpw_time, eqlbpqw_time],
            "Peak memory (MB)": [qlbpw_memory, eqlbpqw_memory],
        },
        title="QLBPW vs EQLBPW Performance",
        xlabel="Algorithm",
        ylabel="Measurement",
        save_fig=save_fig,
    )

    return results


def run_state_space_experiment(
        grid_sizes=None,
        save_fig=False
):
    if grid_sizes is None:
        grid_sizes = tuple(
            preset["grid_size"] for preset in STATE_SPACE_PRESETS
        )

    results = {
        "grid_sizes": [],
        "possible_states": [],

        "qlbpw_states": [],
        "qlbpw_qtable_memory": [],
        "qlbpw_time": [],
        "qlbpw_memory": [],

        "eqlbpw_network_memory": [],
        "eqlbpw_time": [],
        "eqlbpw_memory": [],
    }

    for grid in grid_sizes:
        preset = next(
            preset for preset in STATE_SPACE_PRESETS
            if preset["grid_size"] == grid
        )
        print(f"\n{'=' * 60}")
        print(f"Running state-space experiment: {grid} x {grid}")
        print(f"{'=' * 60}")

        # -------------------------
        # QLBPW
        # -------------------------
        qlbpw_agent, qlbpw_env, qlbpw_time, qlbpw_memory = \
            _measure_simulation(
                lambda: QLBPW_simulate(grid_size=grid, preset=preset)
            )

        qlbpw_states = len(qlbpw_agent.Q)

        qlbpw_qtable_memory = sum(
            values.nbytes
            for values in qlbpw_agent.Q.values()
        ) / (1024 * 1024)

        # -------------------------
        # EQLBPW
        # -------------------------
        eqlbpqw_agent, eqlbpqw_env, eqlbpqw_time, eqlbpqw_memory = \
            _measure_simulation(
                lambda: EQLBPW_simulate(grid_size=grid, preset=preset)
            )

        parameter_memory, optimizer_memory = \
            eqlbpqw_env.tracker.get_network_memory()

        # -------------------------
        # Store results
        # -------------------------
        results["grid_sizes"].append(grid)
        results["possible_states"].append(grid * grid)

        results["qlbpw_states"].append(qlbpw_states)
        results["qlbpw_qtable_memory"].append(qlbpw_qtable_memory)
        results["qlbpw_time"].append(qlbpw_time)
        results["qlbpw_memory"].append(qlbpw_memory)

        results["eqlbpw_network_memory"].append(
            parameter_memory + optimizer_memory
        )
        results["eqlbpw_time"].append(eqlbpqw_time)
        results["eqlbpw_memory"].append(eqlbpqw_memory)

    return results


def plot_state_space_results(results, save_fig=False):
    visualizer = Visualizer(
        agent=None,
        env=None
    )

    # QLBPW Q-table memory
    visualizer.plot_line(
        x=results["possible_states"],
        y=results["qlbpw_qtable_memory"],
        title="QLBPW Q-Table Memory vs State Space",
        xlabel="Number of Possible States",
        ylabel="Q-Table Memory (MB)",
        save_fig=save_fig
    )

    # Training time
    visualizer.compare_plot_line(
        x=results["grid_sizes"],
        series={
            "QLBPW": results["qlbpw_time"],
            "EQLBPW": results["eqlbpw_time"]
        },
        title="Training Time vs Grid Size",
        xlabel="Grid Size",
        ylabel="Training Time (seconds)",
        save_fig=save_fig
    )

    # Peak memory
    visualizer.compare_plot_line(
        x=results["grid_sizes"],
        series={
            "QLBPW": results["qlbpw_memory"],
            "EQLBPW": results["eqlbpw_memory"]
        },
        title="Peak Python Memory vs Grid Size",
        xlabel="Grid Size",
        ylabel="Peak Memory (MB)",
        save_fig=save_fig
    )

    # Number of states actually represented in Q-table
    visualizer.plot_line(
        x=results["grid_sizes"],
        y=results["qlbpw_states"],
        title="QLBPW Represented States vs Grid Size",
        xlabel="Grid Size",
        ylabel="Number of Q-Table States",
        save_fig=save_fig
    )

    # EQLBPW network memory
    visualizer.plot_line(
        x=results["grid_sizes"],
        y=results["eqlbpw_network_memory"],
        title="EQLBPW Network Memory vs Grid Size",
        xlabel="Grid Size",
        ylabel="Network + Optimizer Memory (MB)",
        save_fig=save_fig
    )


def _greedy_path(agent, env, is_dqn):
    state = env.start_state
    path = [state]
    total_reward = 0
    max_steps = env.grid_rows * env.grid_cols * 2

    for _ in range(max_steps):
        if is_dqn:
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = agent.main_net(state_tensor).argmax(dim=1).item()
        else:
            q_values = agent.Q.get(state, np.zeros(agent.no_of_actions))
            action = int(np.argmax(q_values))

        next_state, reward, is_terminal = env.take_step(state, action)
        path.append(next_state)
        total_reward += reward
        state = next_state
        if is_terminal:
            break

    return path, total_reward


def _bfs_shortest_path(env):
    """Return one shortest valid path using the environment's grid and obstacles."""
    from collections import deque

    start = env.start_state
    goal = env.end_state
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        state, path = queue.popleft()
        if state == goal:
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


def run_local_optimum_experiment(episodes=100, seeds=(0, 1, 2, 3, 4)):
    """
    Test whether QLBPW can converge to a valid but suboptimal route.

    The environment is fixed across runs. Each seed changes only the
    stochastic training process. The BFS shortest path is the ground truth.
    """
    results = []

    # Validate the experimental map before training.
    validation_agent = QLBPWAgent(
        alpha=0.1,
        gamma=0.9,
        beta=0.3,
        e=0.9,
        e_min=0.1,
        e_decay=0.998,
        no_of_states=4,
        no_of_actions=4,
        max_buffer=20,
        batch_size=2000,
    )
    validation_env = QLBPWEnvironment(
        grid=LOCAL_OPTIMUM_PRESET["grid_size"],
        start_state=LOCAL_OPTIMUM_PRESET["start_state"],
        end_state=LOCAL_OPTIMUM_PRESET["end_state"],
        agent=validation_agent,
        episodes=1,
        ep_tracker=1,
        no_of_obstacles=0,
        static_obstacles=LOCAL_OPTIMUM_PRESET["obstacles"],
        is_dynamic_obs=False,
    )
    validation_env.generate_obstacles()

    shortest_path = _bfs_shortest_path(validation_env)
    if shortest_path is None:
        raise ValueError("LOCAL_OPTIMUM_PRESET has no valid path from start to goal.")

    shortest_steps = len(shortest_path) - 1
    print("\n" + "=" * 60)
    print("QLBPW Local-Optimum Experiment")
    print("=" * 60)
    print(f"Grid: {LOCAL_OPTIMUM_PRESET['grid_size']} x {LOCAL_OPTIMUM_PRESET['grid_size']}")
    print(f"Shortest valid path (BFS): {shortest_steps} steps")
    print(f"Training episodes per seed: {episodes}")
    print(f"Seeds: {tuple(seeds)}")

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        agent, env = QLBPW_simulate(
            grid_size=LOCAL_OPTIMUM_PRESET["grid_size"],
            episodes=episodes,
            preset=LOCAL_OPTIMUM_PRESET,
        )

        learned_path, learned_reward = _greedy_path(
            agent,
            env,
            is_dqn=False,
        )

        learned_steps = len(learned_path) - 1
        reached_goal = learned_path[-1] == env.end_state
        is_optimal = reached_goal and learned_steps == shortest_steps
        optimality = (
            shortest_steps / learned_steps
            if reached_goal and learned_steps > 0
            else 0.0
        )

        results.append({
            "seed": seed,
            "goal_reached": reached_goal,
            "learned_steps": learned_steps,
            "shortest_steps": shortest_steps,
            "optimal": is_optimal,
            "optimality": optimality,
            "total_reward": learned_reward,
            "path": learned_path,
        })

        print(
            f"Seed {seed}: "
            f"goal={reached_goal}, "
            f"learned={learned_steps}, "
            f"optimal={is_optimal}, "
            f"optimality={optimality:.3f}, "
            f"reward={learned_reward}"
        )

    successful_runs = [result for result in results if result["goal_reached"]]
    suboptimal_runs = [
        result for result in successful_runs
        if not result["optimal"]
    ]

    success_rate = (
        len(successful_runs) / len(results)
        if results else 0.0
    )
    optimal_solution_rate = (
        sum(result["optimal"] for result in successful_runs) / len(successful_runs)
        if successful_runs else 0.0
    )
    suboptimal_solution_rate = (
        len(suboptimal_runs) / len(successful_runs)
        if successful_runs else 0.0
    )

    print("\nSummary")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Optimal-solution rate: {optimal_solution_rate:.2%}")
    print(f"Suboptimal-solution rate: {suboptimal_solution_rate:.2%}")

    return {
        "preset": LOCAL_OPTIMUM_PRESET,
        "shortest_path": shortest_path,
        "shortest_steps": shortest_steps,
        "runs": results,
        "success_rate": success_rate,
        "optimal_solution_rate": optimal_solution_rate,
        "suboptimal_solution_rate": suboptimal_solution_rate,
    }


def run_comparison(save_fig=False):
    qlbpw_agent, qlbpw_env = QLBPW_simulate()
    eqlbpqw_agent, eqlbpqw_env = EQLBPW_simulate()

    qlbpw_path, qlbpw_reward = _greedy_path(qlbpw_agent, qlbpw_env, is_dqn=False)
    eqlbpqw_path, eqlbpqw_reward = _greedy_path(eqlbpqw_agent, eqlbpqw_env, is_dqn=True)

    visualizer = Visualizer(agent=eqlbpqw_agent, env=eqlbpqw_env)
    visualizer.compare_plot_line(
        x=["QLBPW", "EQLBPW"],
        series={
            "Path length": [len(qlbpw_path) - 1, len(eqlbpqw_path) - 1],
            "Total reward": [qlbpw_reward, eqlbpqw_reward],
        },
        title="QLBPW vs EQLBPW",
        xlabel="Algorithm",
        ylabel="Value",
        save_fig=save_fig,
    )

    return {
        "QLBPW": (qlbpw_agent, qlbpw_env, qlbpw_path),
        "EQLBPW": (eqlbpqw_agent, eqlbpqw_env, eqlbpqw_path),
    }


if __name__ == "__main__":
    results = run_state_space_experiment(
        save_fig=True
    )

    plot_state_space_results(
        results,
        save_fig=True
    )