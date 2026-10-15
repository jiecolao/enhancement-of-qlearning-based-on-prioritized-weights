"""Interactive Streamlit dashboard for QLBPW and EQLBPW path-planning experiments.

Run from the repository root with:
    python -m streamlit run src/app.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Support both `streamlit run src/app.py` from the repository root and
# `streamlit run app.py` from inside src/.
SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_config import EQLBPWAgent, QLBPWAgent
from env_config import GeneralEnvironment, Intramuros
from env_settings import PRESET_ENVIRONMENTS, OBSTACLES
from QLBPW.agent import Agent as QLBPWAgentImpl
from QLBPW.environment import Environment as QLBPWEnvironment
from src.QLBPW.simulator import simulate as simulate_qlbpw
from EQLBPW.agent import Agent as EQLBPWAgentImpl
from EQLBPW.environment import Environment as EQLBPWEnvironment
from src.EQLBPW.simulator import simulate as simulate_eqlbpw


st.set_page_config(page_title="Prioritized Q-Learning Lab", page_icon="🧭", layout="wide")
st.title("🧭 Prioritized Q-Learning Lab")
st.caption("Configure, train, and inspect grid-navigation agents.")

ALGORITHM_INFO = {
    "QLBPW": "Tabular Q-learning with rank-based prioritized experience replay.",
    "EQLBPW": "Deep Double Q-learning with prioritized replay and path-aware weights.",
}


def coordinate_label(name: str, coordinate: tuple[int, int]) -> str:
    return f"{name.replace('_', ' ').title()} — {coordinate}"


def get_states(preset: dict, key: str) -> dict[str, tuple[int, int]]:
    value = preset[key]
    if isinstance(value, dict):
        return {str(name): tuple(coord) for name, coord in value.items()}
    return {preset["name"]: tuple(value)}


def draw_grid(size: int, start: tuple[int, int], goal: tuple[int, int],
              obstacles: set[tuple[int, int]], title: str,
              path: list[tuple[int, int]] | None = None, target=None) -> None:
    # Render larger maps compactly while retaining the grid and coordinates.
    fig, ax = plt.subplots(figsize=(min(9, max(4, size * 0.16)),
                                    min(8, max(4, size * 0.16))))
    cells = np.zeros((size, size, 3), dtype=float)
    cells[:, :] = (0.96, 0.97, 0.98)
    for x, y in obstacles:
        if 0 <= x < size and 0 <= y < size:
            cells[y, x] = (0.18, 0.22, 0.29)
    sx, sy = start
    gx, gy = goal
    cells[sy, sx] = (0.16, 0.68, 0.42)
    cells[gy, gx] = (0.95, 0.68, 0.20)
    ax.imshow(cells, origin="upper", interpolation="nearest")
    if path:
        px, py = zip(*path)
        ax.plot(px, py, color="#2563eb", linewidth=max(1.5, 5 / max(1, size / 8)),
                marker="o", markersize=max(2, 5 / max(1, size / 10)),
                alpha=0.9, zorder=4, label="Learned greedy path")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_xticks(np.arange(-0.5, size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, size, 1), minor=True)
    ax.grid(which="minor", color="#cbd5e1", linewidth=0.45)
    tick_step = max(1, size // 10)
    ax.set_xticks(np.arange(0, size, tick_step))
    ax.set_yticks(np.arange(0, size, tick_step))
    ax.tick_params(length=0, labelsize=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.text(sx, sy, "S", ha="center", va="center", color="white",
            fontweight="bold", fontsize=max(6, min(11, 150 / size)))
    ax.text(gx, gy, "G", ha="center", va="center", color="#1f2937",
            fontweight="bold", fontsize=max(6, min(11, 150 / size)))
    fig.tight_layout()
    if target is None:
        st.pyplot(fig, use_container_width=True)
    else:
        target.pyplot(fig, use_container_width=True)
    plt.close(fig)



def learned_path(agent, env, algorithm: str) -> list[tuple[int, int]]:
    """Roll out the current greedy policy without exploration."""
    path = [tuple(env.start_state)]
    state = tuple(env.start_state)
    original_pos, original_steps = env.agent_pos, env.steps
    was_training = getattr(getattr(agent, "main_net", None), "training", None)
    try:
        for _ in range(env.max_steps):
            env.agent_pos = state
            if algorithm == "QLBPW":
                values = agent.Q.get(state, np.zeros(agent.no_of_actions))
                action = int(np.argmax(values))
            else:
                import torch
                agent.main_net.eval()
                with torch.no_grad():
                    obs = torch.as_tensor(env.get_state(), dtype=torch.float32).unsqueeze(0)
                    action = int(agent.main_net(obs).argmax(dim=1).item())
            next_state, _, done, *rest = env.take_step(state, action)
            state = tuple(next_state)
            path.append(state)
            if done or state == tuple(env.end_state):
                break
    finally:
        env.agent_pos, env.steps = original_pos, original_steps
        if algorithm == "EQLBPW" and was_training is not None:
            agent.main_net.train(was_training)
    return path


def render_metrics(tracker, target, window=50):
    """Plot reward, rolling success rate, and steps per episode."""
    if tracker is None or not tracker.reward_history:
        target.info("Metrics will appear after the first completed episode.")
        return
    rewards = np.asarray(tracker.reward_history, dtype=float)
    steps = np.asarray(tracker.steps_history, dtype=float)
    successes = np.asarray(tracker.success_history, dtype=float)
    episodes = np.arange(1, len(rewards) + 1)
    span = min(window, len(rewards))
    mean_reward = np.convolve(rewards, np.ones(span) / span, mode="valid")
    mean_steps = np.convolve(steps, np.ones(span) / span, mode="valid")
    success_rate = np.convolve(successes, np.ones(span) / span, mode="valid") * 100
    rolling_x = episodes[len(episodes) - len(mean_reward):]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(episodes, rewards, alpha=0.25, linewidth=0.8, label="Episode reward")
    axes[0].plot(rolling_x, mean_reward, linewidth=2, label=f"{span}-episode mean")
    axes[0].set_ylabel("Reward")
    axes[0].legend(loc="best")
    axes[1].plot(rolling_x, success_rate, linewidth=2)
    axes[1].set_ylabel(f"Success (%)\n({span}-episode window)")
    axes[1].set_ylim(-3, 103)
    axes[2].plot(episodes, steps, alpha=0.25, linewidth=0.8, label="Steps / episode")
    axes[2].plot(rolling_x, mean_steps, linewidth=2, label=f"{span}-episode mean")
    axes[2].set_ylabel("Steps")
    axes[2].set_xlabel("Episode")
    axes[2].legend(loc="best")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    target.pyplot(fig, use_container_width=True)
    plt.close(fig)


class LiveLog(io.StringIO):
    """Capture stdout and periodically refresh a Streamlit terminal panel."""
    def __init__(self, placeholder, refresh_chars: int = 300):
        super().__init__()
        self.placeholder = placeholder
        self.refresh_chars = refresh_chars
        self.last_render = 0

    def write(self, text: str) -> int:
        result = super().write(text)
        current_len = self.tell()
        if "\n" in text or current_len - self.last_render >= self.refresh_chars:
            self.placeholder.code(self.getvalue()[-18000:] or "Waiting for training output…",
                                  language="text")
            self.last_render = current_len
        return result

    def flush(self) -> None:
        self.placeholder.code(self.getvalue()[-18000:] or "Waiting for training output…",
                              language="text")


def render_history(algorithm: str) -> None:
    log_dir = SRC_DIR / algorithm / "training_logs"
    st.subheader("Training history")
    if not log_dir.exists():
        st.info("No saved training logs yet. Start a run to create one.")
        return
    logs = sorted(log_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        st.info("No saved training logs yet. Start a run to create one.")
        return
    selected = st.selectbox(
        "Previous log",
        logs,
        format_func=lambda p: f"{p.name} · {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}",
        key=f"{algorithm}_history_log",
    )
    try:
        text = selected.read_text(encoding="utf-8", errors="replace")
        st.text_area("Log contents", value=text, height=260, key=f"{algorithm}_history_text")
        st.download_button("Download selected log", data=text,
                           file_name=selected.name, mime="text/plain",
                           key=f"{algorithm}_download_log")
    except OSError as exc:
        st.error(f"Could not read this log: {exc}")


def algorithm_page(algorithm: str) -> None:
    st.header(f"{algorithm} simulator")
    st.write(ALGORITHM_INFO[algorithm])
    preset_by_name = {item["name"]: item for item in PRESET_ENVIRONMENTS}
    preset_names = list(preset_by_name)
    default_name = next((name for name in preset_names if name.startswith("Intramuros")),
                        preset_names[0])
    with st.sidebar:
        st.subheader(f"{algorithm} setup")
        environment_name = st.selectbox("Environment preset", preset_names,
                                        index=preset_names.index(default_name),
                                        key=f"{algorithm}_environment")
        preset = preset_by_name[environment_name]
        size = int(preset["grid_size"])
        starts = get_states(preset, "start_state")
        goals = get_states(preset, "end_state")
        start_names = list(starts)
        goal_names = list(goals)
        if isinstance(preset["start_state"], dict):
            start_name = st.selectbox("Start state", start_names,
                                      index=start_names.index("fort_santiago") if "fort_santiago" in start_names else 0,
                                      format_func=lambda n: coordinate_label(n, starts[n]),
                                      key=f"{algorithm}_start")
            goal_name = st.selectbox("Goal state", goal_names,
                                     index=goal_names.index("enter_exit4") if "enter_exit4" in goal_names else len(goal_names)-1,
                                     format_func=lambda n: coordinate_label(n, goals[n]),
                                     key=f"{algorithm}_goal")
            start, goal = starts[start_name], goals[goal_name]
        else:
            start, goal = starts[start_names[0]], goals[goal_names[0]]

        st.caption(f"Grid: {size} × {size}")
        episodes = st.number_input("Training episodes", min_value=1, max_value=100000,
                                   value=1000 if algorithm == "QLBPW" else 1000,
                                   step=100, key=f"{algorithm}_episodes")
        report_every = st.number_input("Log interval (episodes)", min_value=1,
                                       max_value=max(1, int(episodes)), value=min(10, int(episodes)),
                                       key=f"{algorithm}_report_every")
        dynamic = st.toggle("Enable dynamic obstacles", value=False,
                            key=f"{algorithm}_dynamic")
        dynamic_count = 0
        if dynamic:
            dynamic_count = st.number_input("Dynamic obstacle count", min_value=1,
                                            max_value=max(1, size * size - 2),
                                            value=min(5, max(1, size * size - 2)),
                                            key=f"{algorithm}_dynamic_count")
            st.caption("Obstacles are regenerated every 10 episodes; start and goal are excluded.")

        with st.expander("Agent parameters", expanded=True):
            if algorithm == "QLBPW":
                defaults = asdict(QLBPWAgent())
                agent_params = {
                    "alpha": st.number_input("Learning rate (alpha)", 0.0001, 1.0,
                                              float(defaults["alpha"]), 0.01, format="%.4f", key="q_alpha"),
                    "gamma": st.number_input("Discount factor (gamma)", 0.0, 1.0,
                                              float(defaults["gamma"]), 0.01, key="q_gamma"),
                    "beta": st.number_input("Priority exponent (beta)", 0.0, 1.0,
                                             float(defaults["beta"]), 0.05, key="q_beta"),
                    "e": st.number_input("Initial epsilon", 0.0, 1.0,
                                          float(defaults["e"]), 0.05, key="q_e"),
                    "no_of_actions": 4,
                    "max_buffer": int(st.number_input("Replay capacity", 1, 1000000,
                                                       int(defaults["max_buffer"]), 100, key="q_buffer")),
                    "batch_size": int(st.number_input("Batch size", 1, 10000,
                                                       int(defaults["batch_size"]), 1, key="q_batch")),
                }
            else:
                defaults = asdict(EQLBPWAgent())
                agent_params = {
                    "state_dim": 29,
                    "action_dim": 4,
                    "learning_rate": st.number_input("Learning rate", 0.000001, 1.0,
                                                     float(defaults["learning_rate"]), format="%.5f", key="e_lr"),
                    "gamma": st.number_input("Discount factor (gamma)", 0.0, 1.0,
                                             float(defaults["gamma"]), 0.01, key="e_gamma"),
                    "priority_alpha": st.number_input("Priority alpha", 0.0, 1.0,
                                                      float(defaults["priority_alpha"]), 0.05, key="e_pa"),
                    "beta_start": st.number_input("Importance beta start", 0.0, 1.0,
                                                  float(defaults["beta_start"]), 0.05, key="e_bs"),
                    "beta_end": st.number_input("Importance beta end", 0.0, 1.0,
                                                float(defaults["beta_end"]), 0.05, key="e_be"),
                    "e": st.number_input("Initial epsilon", 0.0, 1.0,
                                          float(defaults["e"]), 0.05, key="e_e"),
                    "e_min": st.number_input("Minimum epsilon", 0.0, 1.0,
                                              float(defaults["e_min"]), 0.01, key="e_emin"),
                    "e_decay": st.number_input("Epsilon decay", 0.8, 1.0,
                                               float(defaults["e_decay"]), 0.001, format="%.3f", key="e_decay"),
                    "max_buffer": int(st.number_input("Replay capacity", 1, 1000000,
                                                       int(defaults["max_buffer"]), 100, key="e_buffer")),
                    "batch_size": int(st.number_input("Batch size", 1, 10000,
                                                       int(defaults["batch_size"]), 8, key="e_batch")),
                    "target_sync_freq": int(st.number_input("Target sync frequency", 1, 10000,
                                                            int(defaults["target_sync_freq"]), 1, key="e_sync")),
                    "collision_weight": st.number_input("Collision priority weight", 0.0, 100.0,
                                                        float(defaults["collision_weight"]), 0.25, key="e_cw"),
                    "goal_weight": st.number_input("Goal priority weight", 0.0, 100.0,
                                                   float(defaults["goal_weight"]), 0.25, key="e_gw"),
                    "distance_weight": st.number_input("Distance priority weight", 0.0, 100.0,
                                                       float(defaults["distance_weight"]), 0.1, key="e_dw"),
                }

        st.caption("Values are initialized from the repository's agent/environment config defaults.")
        run_clicked = st.button(f"▶ Run {algorithm} training", type="primary",
                                use_container_width=True, key=f"{algorithm}_run")

    # The repository presets are the source of truth for map dimensions and static obstacles.
    static_obstacles = set(preset.get("obstacles", set()))
    # env_settings.py keeps detailed obstacle layouts separately from presets.
    obstacle_preset = next(
        (item for item in OBSTACLES if item.get("name") == environment_name),
        None,
    )
    if obstacle_preset is None and environment_name == "QLBPW (9x9)":
        obstacle_preset = next((item for item in OBSTACLES if item.get("name") == "QLBPW"), None)
    if obstacle_preset is not None:
        static_obstacles = set(obstacle_preset.get("obstacles", static_obstacles))
    static_obstacles.discard(start)
    static_obstacles.discard(goal)
    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("Environment map")
        map_slot = st.empty()
        draw_grid(size, start, goal, static_obstacles, environment_name, target=map_slot)
        st.caption("S = start · G = goal · dark cells = obstacles · blue line = learned greedy path")
    with right:
        st.subheader("Run configuration")
        st.write({"Algorithm": algorithm, "Environment": environment_name,
                  "Grid": f"{size} × {size}", "Start": start, "Goal": goal,
                  "Episodes": int(episodes), "Dynamic obstacles": bool(dynamic),
                  "Dynamic obstacle count": int(dynamic_count) if dynamic else 0,
                  "Static obstacles": len(static_obstacles)})
        st.subheader("Live training terminal")
        with st.container(height=320, border=True):
            terminal = st.empty()
            terminal.code("Training output will appear here when a run starts.", language="text")

    if run_clicked:
        if start == goal:
            st.error("Start and goal must be different states.")
        elif int(agent_params.get("batch_size", 1)) > int(agent_params.get("max_buffer", 1)):
            st.error("Batch size cannot exceed replay capacity.")
        elif algorithm == "EQLBPW" and agent_params["beta_start"] > agent_params["beta_end"]:
            st.error("Importance beta start must be less than or equal to beta end.")
        else:
            run_button_placeholder = st.empty()
            run_button_placeholder.info("Training is running. Keep this page open to watch the log.")
            log_capture = LiveLog(terminal)
            metrics_slot = st.empty()
            started = time.time()
            try:
                # Construct with the selected environment and config-driven values.
                env_params = {
                    "grid": size,
                    "start_state": start,
                    "end_state": goal,
                    "episodes": int(episodes),
                    "ep_tracker": int(report_every),
                    "no_of_obstacles": int(dynamic_count) if dynamic else 0,
                    "is_dynamic_obs": bool(dynamic),
                    "static_obstacles": static_obstacles,
                }
                if algorithm == "QLBPW":
                    agent = QLBPWAgentImpl(**agent_params)
                    env = QLBPWEnvironment(agent=agent, **env_params)
                    simulation = simulate_qlbpw
                else:
                    agent = EQLBPWAgentImpl(**agent_params)
                    env = EQLBPWEnvironment(agent=agent, **env_params)
                    simulation = simulate_eqlbpw
                with contextlib.redirect_stdout(log_capture), contextlib.redirect_stderr(log_capture):
                    # Hook episode recording to refresh the map and charts periodically.
                    original_record_episode = env.tracker.record_episode
                    episode_counter = {"n": 0}
                    def dashboard_record_episode(success):
                        original_record_episode(success)
                        episode_counter["n"] += 1
                        n = episode_counter["n"]
                        if n % int(report_every) == 0 or n == int(episodes):
                            current_path = learned_path(agent, env, algorithm)
                            draw_grid(size, start, goal, set(env.obstacles),
                                      environment_name + " · current policy",
                                      path=current_path, target=map_slot)
                            with metrics_slot.container():
                                st.subheader("Learning progress")
                                render_metrics(env.tracker, st.empty())
                    env.tracker.record_episode = dashboard_record_episode
                    simulation(agent=agent, env=env)
                    final_path = learned_path(agent, env, algorithm)
                    draw_grid(size, start, goal, set(env.obstacles) or static_obstacles,
                              environment_name + " · final learned policy",
                              path=final_path, target=map_slot)
                    with metrics_slot.container():
                        st.subheader("Learning progress")
                        render_metrics(env.tracker, st.empty())
                    log_capture.write(f"\nDashboard run completed in {time.time() - started:.1f}s.\n")
                    log_capture.flush()
                st.success(f"{algorithm} training completed in {time.time() - started:.1f} seconds.")
                st.session_state[f"{algorithm}_last_run"] = {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "environment": environment_name,
                    "episodes": int(episodes),
                    "log": log_capture.getvalue(),
                    "rewards": list(getattr(env.tracker, "reward_history", [])),
                    "successes": list(getattr(env.tracker, "success_history", [])),
                    "steps": list(getattr(env.tracker, "steps_history", [])),
                    "path": final_path,
                }
            except Exception as exc:
                log_capture.write(f"\nERROR: {type(exc).__name__}: {exc}\n")
                log_capture.flush()
                st.exception(exc)
            finally:
                run_button_placeholder.empty()

    previous = st.session_state.get(f"{algorithm}_last_run")
    if previous:
        with st.expander("Most recent dashboard run", expanded=False):
            st.write(f"{previous['time']} · {previous['environment']} · {previous['episodes']} episodes")
            st.code(previous["log"][-18000:], language="text")
            st.caption("Episode metrics and the final learned path are retained for this session.")

    st.divider()
    render_history(algorithm)


with st.sidebar:
    st.title("Algorithm")
    selected_algorithm = st.radio(
        "Choose a simulator",
        ["QLBPW", "EQLBPW"],
        index=0,
        format_func=lambda name: f"{name} · {'Tabular' if name == 'QLBPW' else 'Deep RL'}",
    )
    st.divider()
    st.markdown("**Quick guide**")
    st.markdown("1. Select an algorithm and environment.\n2. Adjust agent parameters.\n3. Optionally enable dynamic obstacles.\n4. Run training and inspect the terminal/history.")

algorithm_page(selected_algorithm)
