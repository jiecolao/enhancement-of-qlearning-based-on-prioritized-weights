from EQLBPW.agent import Agent as EQLBPWAgent
from EQLBPW.environment import Environment as EQLBPWEnvironment
from EQLBPW.simulator import simulate as EQLBPW_simulate
from QLBPW.agent import Agent as QLBPWAgent
from QLBPW.environment import Environment as QLBPWEnvironment
from QLBPW.simulator import simulate as QLBPW_simulate
from visualizer import Visualizer
import numpy as np
import torch


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


if __name__ == "__main__":
    run_comparison(save_fig=True)