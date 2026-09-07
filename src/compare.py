from .EQLBPW.agent import Agent as EQLBPWAgent
from .EQLBPW.environment import Environment as EQLBPWEnvironment
from .EQLBPW_simulator import simulate as EQLBPW_simulate
from .QLBPW.agent import Agent as QLBPWAgent
from .QLBPW.environment import Environment as QLBPWEnvironment
from .QLBPW_simulator import simulate as QLBPW_simulate
from .visualizer import Visualizer
from .env_settings import OBSTACLES, PRESET_ENVIRONMENTS
import json
from pathlib import Path
import numpy as np
import torch
import time
import tracemalloc
import random

STATE_SPACE_PRESETS = [
    # PRESET_ENVIRONMENTS[3],
    PRESET_ENVIRONMENTS[4],
    PRESET_ENVIRONMENTS[5],
    PRESET_ENVIRONMENTS[6],
    # PRESET_ENVIRONMENTS[7],
    # PRESET_ENVIRONMENTS[8],
]


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

if __name__ == "__main__":
    results = run_state_space_experiment(
        # grid_sizes=(20, 20),
        save_fig=True
    )

    output_path = Path(__file__).resolve().parent / "datas" / "compare_state_space_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    plot_state_space_results(
        results,
        save_fig=True
    )