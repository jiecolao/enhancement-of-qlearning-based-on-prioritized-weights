import random
import numpy as np
import pandas as pd
import streamlit as st
import torch

from QLBPW.agent import Agent as QLBPWAgent
from QLBPW.environment import Environment as QLBPWEnvironment
from EQLBPW.agent import Agent as EQLBPWAgent
from EQLBPW.environment import Environment as EQLBPWEnvironment
from env_settings import PRESET_ENVIRONMENTS


st.set_page_config(page_title="Overestimation Comparison", layout="wide")
st.title("QLBPW vs EQLBPW: Overestimation")
st.caption(
    "Compares the estimated value at the start state with the discounted return actually obtained by the greedy policy."
)


def resolve_preset(preset):
    start = preset["start_state"]
    goal = preset["end_state"]
    if isinstance(start, dict):
        start = start["fort_santiago"]
    if isinstance(goal, dict):
        goal = goal["enter_exit4"]
    return {**preset, "start_state": start, "end_state": goal}


def build_qlbpw(preset, episodes, alpha, gamma, epsilon, buffer_size, batch_size, dynamic, dynamic_count):
    agent = QLBPWAgent(
        alpha=alpha,
        gamma=gamma,
        beta=0.3,
        e=epsilon,
        no_of_actions=4,
        max_buffer=buffer_size,
        batch_size=batch_size,
    )
    env = QLBPWEnvironment(
        grid=preset["grid_size"],
        start_state=preset["start_state"],
        end_state=preset["end_state"],
        agent=agent,
        episodes=episodes,
        ep_tracker=episodes + 1,
        no_of_obstacles=dynamic_count,
        static_obstacles=preset["obstacles"],
        is_dynamic_obs=dynamic,
    )
    env.generate_obstacles()
    for episode in range(episodes):
        env.agent_pos = env.start_state
        env.steps = 0
        terminal = False
        steps = 0
        while not terminal and steps < env.max_steps:
            state = env.agent_pos
            action = agent.epsilon_greedy(state)
            next_state, reward, terminal = env.take_step(state, action)
            if state not in agent.Q:
                agent.Q[state] = np.zeros(agent.no_of_actions)
            current_q = agent.Q[state][action]
            if terminal:
                td_target = reward
            else:
                if next_state not in agent.Q:
                    agent.Q[next_state] = np.zeros(agent.no_of_actions)
                td_target = reward + agent.gamma * np.max(agent.Q[next_state])
            td_error = td_target - current_q
            agent.memory.push(state, action, reward, next_state, td_error)
            if len(agent.memory) > 0:
                sample = agent.adjust_lr()
                agent.update_Q(
                    *sample,
                    end_state=env.end_state,
                    obstacles=env.obstacles,
                )
            env.agent_pos = next_state
            steps += 1
        if dynamic and (episode + 1) % 10 == 0:
            env.generate_obstacles()
    return agent, env


def build_eqlbpw(preset, episodes, alpha, gamma, epsilon, buffer_size, batch_size,
                 priority_alpha, beta_start, beta_end, collision_weight,
                 goal_weight, distance_weight, target_sync, dynamic, dynamic_count):
    agent = EQLBPWAgent(
        state_dim=29,
        action_dim=4,
        learning_rate=alpha,
        gamma=gamma,
        priority_alpha=priority_alpha,
        beta_start=beta_start,
        beta_end=beta_end,
        e=epsilon,
        e_min=0.05,
        e_decay=0.995,
        max_buffer=buffer_size,
        batch_size=batch_size,
        target_sync_freq=target_sync,
        collision_weight=collision_weight,
        goal_weight=goal_weight,
        distance_weight=distance_weight,
    )
    env = EQLBPWEnvironment(
        grid=preset["grid_size"],
        start_state=preset["start_state"],
        end_state=preset["end_state"],
        agent=agent,
        episodes=episodes,
        ep_tracker=episodes + 1,
        no_of_obstacles=dynamic_count,
        static_obstacles=preset["obstacles"],
        is_dynamic_obs=dynamic,
    )
    env.generate_obstacles()
    for episode in range(episodes):
        state = env.reset()
        terminal = False
        steps = 0
        while not terminal and steps < env.max_steps:
            action = agent.e_greedy(state)
            next_position, reward, terminal, info = env.take_step(env.agent_pos, action)
            next_state = env.get_state()
            agent.memory.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=terminal,
                collision=info["collision"],
                goal=info["goal"],
                distance_progress=info["distance_progress"],
            )
            agent.update()
            state = next_state
            steps += 1
        progress = episode / max(episodes - 1, 1)
        agent.update_beta(progress)
        agent.decay_e()
        if (episode + 1) % agent.target_sync_freq == 0:
            agent.sync_target()
        if dynamic and (episode + 1) % 10 == 0:
            env.generate_obstacles()
    return agent, env


def evaluate_qlbpw(agent, env, trials, gamma, dynamic):
    rows = []
    for _ in range(trials):
        if dynamic:
            env.generate_obstacles()
        state = env.start_state
        env.agent_pos = state
        env.steps = 0
        q_values = agent.Q.get(state, np.zeros(agent.no_of_actions))
        action = int(np.argmax(q_values))
        estimated = float(q_values[action])
        actual = 0.0
        discount = 1.0
        terminal = False
        for _ in range(env.max_steps):
            next_state, reward, terminal = env.take_step(state, action)
            actual += discount * reward
            if terminal:
                break
            discount *= gamma
            state = next_state
            q_values = agent.Q.get(state, np.zeros(agent.no_of_actions))
            action = int(np.argmax(q_values))
        rows.append({"estimated": estimated, "actual": actual, "bias": estimated - actual})
    return pd.DataFrame(rows)


def evaluate_eqlbpw(agent, env, trials, gamma, dynamic):
    rows = []
    was_training = agent.main_net.training
    agent.main_net.eval()
    try:
        with torch.no_grad():
            for _ in range(trials):
                if dynamic:
                    env.generate_obstacles()
                env.agent_pos = env.start_state
                env.steps = 0
                state_features = torch.as_tensor(env.get_state(), dtype=torch.float32).unsqueeze(0)
                q_values = agent.main_net(state_features).squeeze(0)
                action = int(torch.argmax(q_values).item())
                estimated = float(q_values[action].item())
                actual = 0.0
                discount = 1.0
                terminal = False
                for _ in range(env.max_steps):
                    next_state, reward, terminal, _ = env.take_step(env.agent_pos, action)
                    actual += discount * reward
                    if terminal:
                        break
                    discount *= gamma
                    state_features = torch.as_tensor(env.get_state(), dtype=torch.float32).unsqueeze(0)
                    q_values = agent.main_net(state_features).squeeze(0)
                    action = int(torch.argmax(q_values).item())
                rows.append({"estimated": estimated, "actual": actual, "bias": estimated - actual})
    finally:
        agent.main_net.train(was_training)
    return pd.DataFrame(rows)


preset_name = st.selectbox("Environment preset", [p["name"] for p in PRESET_ENVIRONMENTS], index=1)
preset = resolve_preset(next(p for p in PRESET_ENVIRONMENTS if p["name"] == preset_name))

col1, col2, col3 = st.columns(3)
with col1:
    episodes = st.number_input("Training episodes", 10, 5000, 500, step=50)
    trials = st.number_input("Evaluation trials", 1, 200, 30, step=5)
with col2:
    dynamic = st.checkbox("Dynamic obstacles", value=False)
    dynamic_count = st.number_input("Dynamic obstacle count", 0, 30, 5 if dynamic else 0)
with col3:
    seed = st.number_input("Random seed", 0, 999999, 42)
    gamma = st.number_input("Discount factor", 0.5, 0.99, 0.95, step=0.01)

st.info(
    "The bias is estimated Q-value minus discounted return. Positive values indicate overestimation. "
    "QLBPW and EQLBPW use their own training reward scales; the comparison is therefore within-algorithm "
    "and should be interpreted through the bias gap rather than raw Q-value magnitude alone."
)

if st.button("Run comparison", type="primary", use_container_width=True):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    with st.spinner("Training QLBPW and EQLBPW..."):
        q_agent, q_env = build_qlbpw(
            preset, int(episodes), 0.1, gamma, 0.9, 2000, 20,
            dynamic, int(dynamic_count)
        )
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        e_agent, e_env = build_eqlbpw(
            preset, int(episodes), 0.0005, gamma, 1.0, 50000, 64,
            0.6, 0.4, 1.0, 1.0, 2.0, 0.5, 20,
            dynamic, int(dynamic_count)
        )

    q_results = evaluate_qlbpw(q_agent, q_env, int(trials), gamma, dynamic)
    e_results = evaluate_eqlbpw(e_agent, e_env, int(trials), gamma, dynamic)

    comparison = pd.DataFrame({
        "Algorithm": ["QLBPW", "EQLBPW"],
        "Mean estimated Q": [q_results.estimated.mean(), e_results.estimated.mean()],
        "Mean actual return": [q_results.actual.mean(), e_results.actual.mean()],
        "Mean overestimation gap": [q_results.bias.mean(), e_results.bias.mean()],
        "Overestimation frequency (%)": [
            100 * (q_results.bias > 0).mean(),
            100 * (e_results.bias > 0).mean(),
        ],
    })

    st.subheader("Estimated Q-value vs actual return")
    st.bar_chart(comparison.set_index("Algorithm")[["Mean estimated Q", "Mean actual return"]])

    st.subheader("Overestimation gap")
    st.bar_chart(comparison.set_index("Algorithm")[["Mean overestimation gap"]])

    st.subheader("Summary")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    q_bias = q_results.bias.mean()
    e_bias = e_results.bias.mean()
    st.metric(
        "EQLBPW overestimation gap improvement",
        f"{q_bias - e_bias:.4f}",
        delta=f"{(q_bias - e_bias) / max(abs(q_bias), 1e-8) * 100:.1f}% vs QLBPW",
    )
