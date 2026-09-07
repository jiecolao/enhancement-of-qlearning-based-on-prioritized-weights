import random

import numpy as np
import pandas as pd
import streamlit as st
import torch

from EQLBPW.agent import Agent as EQLBPWAgent


st.set_page_config(page_title="Overestimation Benchmark", layout="wide")
st.title("QLBPW vs EQLBPW: Controlled Overestimation Test")
st.caption("A stochastic two-stage MDP with a known true value, designed to expose max-Q overestimation.")

STATE_DIM = 29
ACTION_DIM = 4
ROOT = np.zeros(STATE_DIM, dtype=np.float32)
DECISION = np.zeros(STATE_DIM, dtype=np.float32)
ROOT[0] = 1.0
DECISION[1] = 1.0


def train_qlbpw(reward_matrix, alpha, gamma, epsilon):
    """Clean tabular Q-learning baseline for the overestimation experiment."""
    q_root = np.zeros(ACTION_DIM)
    q_decision = np.zeros(ACTION_DIM)

    for rewards in reward_matrix:
        root_action = random.randrange(ACTION_DIM) if random.random() < epsilon else int(np.argmax(q_root))
        q_root[root_action] += alpha * (gamma * np.max(q_decision) - q_root[root_action])

        decision_action = random.randrange(ACTION_DIM) if random.random() < epsilon else int(np.argmax(q_decision))
        q_decision[decision_action] += alpha * (float(rewards[decision_action]) - q_decision[decision_action])

    return q_root


def train_eqlbpw(reward_matrix, learning_rate, gamma, batch_size, target_sync, priority_alpha, beta_start, beta_end):
    """Train the repository's actual EQLBPW Double-DQN agent on the same MDP."""
    agent = EQLBPWAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        learning_rate=learning_rate,
        gamma=gamma,
        priority_alpha=priority_alpha,
        beta_start=beta_start,
        beta_end=beta_end,
        e=1.0,
        e_min=0.05,
        e_decay=0.995,
        max_buffer=max(5000, batch_size * 20),
        batch_size=batch_size,
        target_sync_freq=target_sync,
        collision_weight=0.0,
        goal_weight=0.0,
        distance_weight=0.0,
    )

    terminal = np.zeros(STATE_DIM, dtype=np.float32)

    for episode, rewards in enumerate(reward_matrix):
        root_action = agent.e_greedy(ROOT)
        agent.memory.push(ROOT, root_action, 0.0, DECISION, False, 0.0, 0.0, 0.0)
        agent.update()

        decision_action = agent.e_greedy(DECISION)
        reward = float(rewards[decision_action])
        agent.memory.push(DECISION, decision_action, reward, terminal, True, 0.0, 0.0, 0.0)
        agent.update()

        progress = episode / max(len(reward_matrix) - 1, 1)
        agent.update_beta(progress)
        agent.decay_e()
        if (episode + 1) % target_sync == 0:
            agent.sync_target()

    with torch.no_grad():
        root_q = agent.main_net(torch.as_tensor(ROOT).unsqueeze(0)).squeeze(0).cpu().numpy()
    return root_q


def run(seeds, episodes, reward_mean, reward_std, alpha, gamma, epsilon, learning_rate,
        batch_size, target_sync, priority_alpha, beta_start, beta_end):
    true_value = gamma * reward_mean
    rows = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        # Same reward samples are supplied to both algorithms for a fair comparison.
        reward_matrix = rng.normal(reward_mean, reward_std, (episodes, ACTION_DIM))

        random.seed(seed)
        np.random.seed(seed)
        q_root = train_qlbpw(reward_matrix, alpha, gamma, epsilon)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        e_root = train_eqlbpw(
            reward_matrix, learning_rate, gamma, batch_size, target_sync,
            priority_alpha, beta_start, beta_end
        )

        for algorithm, q_values in [("QLBPW", q_root), ("EQLBPW", e_root)]:
            estimate = float(np.max(q_values))
            rows.append({
                "Seed": seed,
                "Algorithm": algorithm,
                "True value": true_value,
                "Estimated max Q": estimate,
                "Bias": estimate - true_value,
                "Positive overestimation": max(estimate - true_value, 0.0),
            })

    return pd.DataFrame(rows), true_value


st.markdown(
    """
### How this test works

- **State 0:** the agent moves to State 1 with zero reward.
- **State 1:** the agent chooses one of four actions.
- Each action gives a noisy reward with a **known expected value**.
- All four actions have the same expected reward, so the true value is known exactly:
  `V* = gamma × reward_mean`.
- Standard Q-learning uses `max Q` when backing up State 0.
- EQLBPW uses your Double-DQN target: the main network selects the action and the target network evaluates it.

This isolates the max-Q overestimation mechanism instead of mixing it with path-planning success/failure.
"""
)

c1, c2, c3 = st.columns(3)
with c1:
    episodes = st.number_input("Training episodes", 200, 10000, 3000, step=100)
    seeds_count = st.number_input("Random seeds", 3, 30, 10, step=1)
with c2:
    reward_mean = st.number_input("True reward mean", -1.0, 1.0, 0.0, step=0.1)
    reward_std = st.number_input("Reward noise (std)", 0.1, 5.0, 2.0, step=0.1)
with c3:
    gamma = st.number_input("Discount factor", 0.5, 0.99, 0.95, step=0.01)
    base_seed = st.number_input("Starting seed", 0, 999999, 42)

with st.expander("Advanced parameters"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        alpha = st.number_input("QLBPW learning rate", 0.01, 1.0, 0.1, step=0.01)
    with c2:
        epsilon = st.number_input("QLBPW epsilon", 0.0, 1.0, 0.1, step=0.05)
    with c3:
        learning_rate = st.number_input("EQLBPW learning rate", 0.00001, 0.01, 0.0005, format="%.5f")
    with c4:
        batch_size = st.number_input("EQLBPW batch size", 8, 128, 32, step=8)
    c5, c6, c7 = st.columns(3)
    with c5:
        target_sync = st.number_input("Target sync frequency", 1, 500, 20, step=1)
    with c6:
        priority_alpha = st.number_input("Priority alpha", 0.0, 1.0, 0.6, step=0.1)
    with c7:
        beta_start = st.number_input("Beta start", 0.0, 1.0, 0.4, step=0.1)
        beta_end = st.number_input("Beta end", 0.0, 1.0, 1.0, step=0.1)

st.info("Positive Bias = overestimation. The true value is known analytically, so we do not use an agent's own trajectory as ground truth.")

if st.button("Run stochastic benchmark", type="primary", use_container_width=True):
    seeds = [int(base_seed) + i for i in range(int(seeds_count))]
    with st.spinner(f"Running {len(seeds)} seeds × {int(episodes)} episodes..."):
        results, true_value = run(
            seeds, int(episodes), float(reward_mean), float(reward_std),
            float(alpha), float(gamma), float(epsilon), float(learning_rate),
            int(batch_size), int(target_sync), float(priority_alpha),
            float(beta_start), float(beta_end)
        )

    summary = results.groupby("Algorithm").agg(
        True_value=("True value", "mean"),
        Estimated_max_Q=("Estimated max Q", "mean"),
        Estimated_Q_std=("Estimated max Q", "std"),
        Mean_bias=("Bias", "mean"),
        Bias_std=("Bias", "std"),
        Mean_positive_overestimation=("Positive overestimation", "mean"),
    ).reset_index()

    st.subheader("Estimated max Q vs true value")
    st.bar_chart(summary.set_index("Algorithm")[["True_value", "Estimated_max_Q"]])

    st.subheader("Overestimation bias")
    st.bar_chart(summary.set_index("Algorithm")[["Mean_bias"]])

    display = summary.rename(columns={
        "True_value": "True value",
        "Estimated_max_Q": "Estimated max Q",
        "Estimated_Q_std": "Estimated Q std",
        "Mean_bias": "Mean bias",
        "Bias_std": "Bias std",
        "Mean_positive_overestimation": "Mean positive overestimation",
    }).round(4)
    st.subheader("Summary across seeds")
    st.dataframe(display, use_container_width=True, hide_index=True)

    q_bias = float(summary.loc[summary.Algorithm == "QLBPW", "Mean_bias"].iloc[0])
    e_bias = float(summary.loc[summary.Algorithm == "EQLBPW", "Mean_bias"].iloc[0])
    if e_bias < q_bias:
        st.success(f"EQLBPW has lower mean bias ({e_bias:.4f}) than QLBPW ({q_bias:.4f}) in this controlled test.")
    else:
        st.warning(f"This run does not show lower EQLBPW bias: QLBPW={q_bias:.4f}, EQLBPW={e_bias:.4f}. Increase seeds/episodes or reward noise before drawing a conclusion.")

    st.caption(f"Ground-truth value = {true_value:.4f}. Report mean ± standard deviation across seeds in the thesis.")
