from .QLBPW.agent import Agent as QLBPWAgent
from .EQLBPW.agent import Agent as EQLBPWAgent
from .QLBPW.environment import Environment as QLBPWEnvironment
from .EQLBPW.environment import Environment as EQLBPWEnvironment
from .QLBPW.simulator import simulate as QLBPWSimulate
from .EQLBPW.simulator import simulate as EQLBPWSimulate
from .agent_config import EQLBPWAgent as EQLBPWAgentConfig
from .agent_config import QLBPWAgent as QLBPWAgentConfig
from .env_settings import PRESET_ENVIRONMENTS
from .env_config import *
from .visualizer import Visualizer
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import numpy as np
import json
import torch
import time
import tracemalloc
import random


class Compare:
    def __init__(
            self,
            qlbpw_agent,
            qlbpw_env,
            eqlbpw_agent,
            eqlbpw_env,
        ):
        self.qlbpw_agent: QLBPWAgent = qlbpw_agent
        self.qlbpw_env: QLBPWEnvironment = qlbpw_env
        self.eqlbpw_agent: EQLBPWAgent = eqlbpw_agent
        self.eqlbpw_env: EQLBPWEnvironment = eqlbpw_env

    def _run_algo(self, simulator):
        tracemalloc.start()
        start_time = time.perf_counter()

        agent, env = simulator()

        elapsed_time = time.perf_counter() - start_time
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return agent, env, elapsed_time, peak_memory / (1024 * 1024)

    def measure_epsilon(self):
        _, qlbpw_env, _, _ = \
            self._run_algo(lambda: QLBPWSimulate(self.qlbpw_agent, self.qlbpw_env))
        _, eqlbpw_env, _, _ = \
            self._run_algo(lambda: EQLBPWSimulate(self.eqlbpw_agent, self.eqlbpw_env))

        qlbpw_epsilon = self.qlbpw_env.tracker.epsilon_history
        eqlbpw_epsilon = self.eqlbpw_env.tracker.epsilon_history

        visual = Visualizer(
            agent=qlbpw_agent, 
            env=qlbpw_env
        ).compare_epsilon(QLBPW_e=qlbpw_epsilon , EQBLPW_e=eqlbpw_epsilon, episodes=qlbpw_env.tracker.ep_history, save_fig=True)

        results = {
            "QLBPW": {
                "epsilon_history": qlbpw_epsilon,
            },
            "EQLBPW": {
                "epsilon_history": eqlbpw_epsilon,
            },
        }

        return results

    def measure_memory_usage(self):
        qlbpw_agent, qlbpw_env, qlbpw_time, qlbpw_memory = self._run_algo(
            lambda: QLBPWSimulate(agent=self.qlbpw_agent, env=self.qlbpw_env)
        )
        eqlbpw_agent, eqlbpw_env, eqlbpw_time, eqlbpw_memory = self._run_algo(
            lambda: EQLBPWSimulate(agent=self.eqlbpw_agent, env=self.eqlbpw_env)
        )

        results = {
            "QLBPW": {
                "elapsed_time": qlbpw_time,
                "memory_usage": qlbpw_memory,
            },
            "EQLBPW": {
                "elapsed_time": eqlbpw_time,
                "memory_usage": eqlbpw_memory,
            }
        }

        # Pass the corrected data from the results dict directly into the visualizer
        visualizer = Visualizer(eqlbpw_agent, eqlbpw_env)
        
        # Using a bar chart since "QLBPW" and "EQLBPW" are distinct categorical algorithms
        visualizer.compare_plot_bar(
            categories=["QLBPW", "EQLBPW"],
            series={
                "Elapsed Time (s)": [results["QLBPW"]["elapsed_time"], results["EQLBPW"]["elapsed_time"]],
                "Peak Memory (MB)": [results["QLBPW"]["memory_usage"], results["EQLBPW"]["memory_usage"]],
            },
            title="QLBPW vs EQLBPW Performance & Memory Usage",
            xlabel="Algorithm",
            ylabel="Measurement Value",
            save_fig=False,
        )

        return results

    def measure_elapsed_time_and_memory_usage(self):
        qlbpw_agent, qlbpw_env, qlbpw_time, qlbpw_memory = self._run_algo(lambda: QLBPWSimulate(agent=self.qlbpw_agent, env=self.qlbpw_env))
        eqlbpw_agent, eqlbpw_env, eqlbpw_time, eqlbpw_memory = self._run_algo(lambda: EQLBPWSimulate(agent=self.eqlbpw_agent, env=self.eqlbpw_env))

        results = {
            "QLBPW": {
                "elapsed_time": qlbpw_time,
                "peak_memory": qlbpw_memory,
            },
            "EQLBPW": {
                "elapsed_time": eqlbpw_time,
                "peak_memory": eqlbpw_memory,
            },
        }

        visualizer = Visualizer(eqlbpw_agent, eqlbpw_env)
        visualizer.compare_plot_line(
            x=["QLBPW", "EQLBPW"],
            series={
                "Elapsed time (seconds)": [qlbpw_time, qlbpw_time],
                "Peak memory (MB)": [eqlbpw_memory, eqlbpw_memory],
            },
            title="QLBPW vs EQLBPW Performance",
            xlabel="Algorithm",
            ylabel="Measurement",
            save_fig=False,
        )

        return results

    @staticmethod
    def _new_environment(environment, agent, grid_size):
        return type(environment)(
            grid=grid_size,
            start_state=environment.start_state,
            end_state=environment.end_state,
            agent=agent,
            episodes=environment.episodes,
            ep_tracker=environment.ep_tracker,
            no_of_obstacles=environment.no_of_obstacles,
            static_obstacles=deepcopy(environment.static_obstacles),
            is_dynamic_obs=environment.is_dynamic_obs,
        )

    def measure_state_space(self, grid_sizes=None):
        STATE_SPACE_PRESETS = [
            PRESET_ENVIRONMENTS[3],
            PRESET_ENVIRONMENTS[4],
            PRESET_ENVIRONMENTS[5],
            # PRESET_ENVIRONMENTS[6],
            # PRESET_ENVIRONMENTS[7],
            # PRESET_ENVIRONMENTS[8],
        ]
        
        if grid_sizes is None:
            grid_sizes = tuple(
                preset["grid_size"] for preset in STATE_SPACE_PRESETS
            )
        else:
            grid_sizes = tuple(grid_sizes)

        available_grids = {
            preset["grid_size"] for preset in STATE_SPACE_PRESETS
        }
        unsupported_grids = set(grid_sizes) - available_grids
        if unsupported_grids:
            raise ValueError(
                "No state-space preset exists for grid size(s): "
                + ", ".join(str(grid) for grid in sorted(unsupported_grids))
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
            print(f"\n{'=' * 60}")
            print(f"Running state-space experiment: {grid} x {grid}")
            print(f"{'=' * 60}")

            qlbpw_agent = deepcopy(self.qlbpw_agent)
            qlbpw_env = self._new_environment(
                self.qlbpw_env, qlbpw_agent, grid
            )
            qlbpw_agent, qlbpw_env, qlbpw_time, qlbpw_memory = self._run_algo(lambda: QLBPWSimulate(
                    agent=qlbpw_agent,
                    env=qlbpw_env,
                )
            )

            qlbpw_states = len(qlbpw_agent.Q)
            qlbpw_qtable_memory = sum(
                values.nbytes for values in qlbpw_agent.Q.values()
            ) / (1024 * 1024)

            eqlbpw_agent = deepcopy(self.eqlbpw_agent)
            eqlbpw_env = self._new_environment(
                self.eqlbpw_env, eqlbpw_agent, grid
            )
            eqlbpw_agent, eqlbpw_env, eqlbpw_time, eqlbpw_memory = \
                self._run_algo(
                    lambda: EQLBPWSimulate(
                        agent=eqlbpw_agent,
                        env=eqlbpw_env,
                    )
                )

            parameter_memory, optimizer_memory = \
                eqlbpw_env.tracker.get_network_memory()

            results["grid_sizes"].append(grid)
            results["possible_states"].append(grid * grid)
            results["qlbpw_states"].append(qlbpw_states)
            results["qlbpw_qtable_memory"].append(qlbpw_qtable_memory)
            results["qlbpw_time"].append(qlbpw_time)
            results["qlbpw_memory"].append(qlbpw_memory)
            results["eqlbpw_network_memory"].append(
                parameter_memory + optimizer_memory
            )
            results["eqlbpw_time"].append(eqlbpw_time)
            results["eqlbpw_memory"].append(eqlbpw_memory)

        visualizer = Visualizer(agent=None, env=None)
        
        visualizer.plot_line(
            x=results["possible_states"],
            y=results["qlbpw_qtable_memory"],
            title="QLBPW Q-Table Memory vs State Space",
            xlabel="Number of Possible States",
            ylabel="Q-Table Memory (MB)",
            save_fig=False,
        )
        visualizer.compare_plot_line(
            x=results["grid_sizes"],
            series={
                "QLBPW": results["qlbpw_time"],
                "EQLBPW": results["eqlbpw_time"],
            },
            title="Training Time vs Grid Size",
            xlabel="Grid Size",
            ylabel="Training Time (seconds)",
            save_fig=False,
        )   
        visualizer.plot_line(
            x=results["grid_sizes"],
            y=results["qlbpw_states"],
            title="QLBPW Represented States vs Grid Size",
            xlabel="Grid Size",
            ylabel="Number of Q-Table States",
            save_fig=False,
        )
        visualizer.plot_line(
            x=results["grid_sizes"],
            y=results["eqlbpw_network_memory"],
            title="EQLBPW Network Memory vs Grid Size",
            xlabel="Grid Size",
            ylabel="Network + Optimizer Memory (MB)",
            save_fig=False,
        )

        return results

    def simulate_intramuros(self):
        qlbpw_agent, qlbpw_env, _, _ = self._run_algo(lambda: QLBPWSimulate(agent=self.qlbpw_agent, env=QLBPWEnvironment(agent=self.qlbpw_agent, **asdict(Intramuros()))))
        eqlbpw_agent, eqlbpw_env, _, _ = self._run_algo(lambda: EQLBPWSimulate(agent=self.eqlbpw_agent, env=EQLBPWEnvironment(agent=self.eqlbpw_agent, **asdict(Intramuros()))))

        visual = Visualizer(agent=eqlbpw_agent, env=eqlbpw_env)
        visual.eqlbpqw_visualize_learned_path(save_fig=True)

        visual = Visualizer(agent=qlbpw_agent, env=qlbpw_env)
        visual.qlbpw_visualize_learned_path(agent=qlbpw_agent, env=qlbpw_env, save_fig=True)


if __name__ == "__main__":
    qlbpw_agent = QLBPWAgent(**asdict(QLBPWAgentConfig()))
    eqlbpw_agent = EQLBPWAgent(**asdict(EQLBPWAgentConfig()))

    qlbpw_custom_env = QLBPWEnvironment(agent=qlbpw_agent, **asdict(GeneralEnvironment()))
    eqlbpw_custom_env = EQLBPWEnvironment(agent=eqlbpw_agent, **asdict(GeneralEnvironment()))
    
    compare = Compare(
        qlbpw_agent=qlbpw_agent,
        qlbpw_env=qlbpw_custom_env,
        eqlbpw_agent=eqlbpw_agent,
        eqlbpw_env=eqlbpw_custom_env
    )

    results = compare.measure_memory_usage()
    
    output_path = Path(__file__).resolve().parent / "data" / "comparison_result.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
