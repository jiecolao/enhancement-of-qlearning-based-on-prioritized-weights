from EQLBPW.agent import Agent as EQLBPWAgent
from EQLBPW.environment import Environment as EQLBPWEnvironment
from EQLBPW.simulator import simulate as EQLBPW_simulate
from QLBPW.agent import Agent as QLBPWAgent
from QLBPW.environment import Environment as QLBPWEnvironment
from QLBPW.simulator import simulate as QLBPW_simulate
from visualizer import Visualizer
from env_settings import OBSTACLES
import numpy as np
import torch
import time
import tracemalloc


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

# def measure_elapsed_time_and_memory_usage():
#     eqlbpqw_agent, eqlbpqw_env = EQLBPW_simulate()
#     qlbpw_agent, qlbpw_env = QLBPW_simulate()

#     # ...


if __name__ == "__main__":
    EQLBPW_agent = EQLBPWAgent(
        alpha=0.001,              # Learning Rate
        gamma=0.95,              # Discount Factor
        beta=0.3,               # Beta
        e=1.0,                  # Epsilon
        e_min=0.05,              # Minimun Epsilon
        e_decay=0.97,          # Epsilon Decay
        no_of_actions=4,        # Actions: 1=up, 2=right, 3=down, 4=left 
        batch_size=64,          # The Number of Experiences To Be Sampled
        max_buffer=10000,        # Max Number of Stored Experiences
        target_sync_freq=5      # When should the Target Network sync
    )

    QLBPW_agent = QLBPWAgent(
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

    EQLBPW_env = EQLBPWEnvironment(
        grid = 20,                                      # Grid Environment gridxgrid
        start_state = (4, 0),                           # Agent Starting Position
        end_state = (16, 7),                           # Finish Line
        agent = EQLBPW_agent,                                  # Agent
        episodes = 20,                                   # Episodes to train
        ep_tracker = 5,                                 # How and when should the tracker print the summary
        no_of_obstacles = 0,                            # Number of obstacles to appear. (To spawn, set is_dynamic_obs to True)
        static_obstacles = OBSTACLES[1]["obstacles"],   # Premade obstacles
        is_dynamic_obs = True,                          # Obstacle Event Trigger
    )

    QLBPW_env = QLBPWEnvironment(
        grid=20,
        start_state=(4, 0),
        end_state=(16, 7),
        agent=QLBPW_agent,
        episodes=20,
        ep_tracker=5,
        no_of_obstacles=0,
        static_obstacles= OBSTACLES[1]["obstacles"],
        is_dynamic_obs=True
    )

    measure_elapsed_time_and_memory_usage(save_fig=True)
    # run_comparison(save_fig=True)