import html
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from QLBPW.agent import Agent as QLBPWAgent
from QLBPW.environment import Environment as QLBPWEnvironment
from env_settings import PRESET_ENVIRONMENTS


st.set_page_config(page_title="Q-Learning Dashboard", layout="wide")


def render_grid(grid_size, obstacles, start, goal, path=None):
	grid = np.zeros((grid_size, grid_size))
	for obstacle in obstacles:
		grid[obstacle[1], obstacle[0]] = 1
	for position in path or []:
		grid[position[1], position[0]] = 2
	grid[start[1], start[0]] = 3
	grid[goal[1], goal[0]] = 4

	cmap = mcolors.ListedColormap(
		["#FFFFFF", "#222222", "#B3E5FC", "#4CAF50", "#F44336"]
	)
	norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
	figure, axis = plt.subplots(figsize=(6, 6))
	axis.imshow(grid, cmap=cmap, norm=norm, origin="upper")
	axis.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
	axis.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
	axis.grid(which="minor", color="#DDDDDD", linestyle="-", linewidth=1)
	axis.tick_params(which="both", bottom=False, left=False,
					 labelbottom=False, labelleft=False)
	return figure


def read_tracker_log(environment):
	try:
		with open(environment.tracker.full_log_path, "r", encoding="utf-8") as log_file:
			return log_file.read()
	except FileNotFoundError:
		return "Waiting for the training tracker log..."


def refresh_tracker_log(log_placeholder, environment):
	log_text = html.escape(read_tracker_log(environment))
	log_placeholder.markdown(
		f"""
		<div style="height: 620px; overflow-y: auto; white-space: pre;
				font-family: monospace; font-size: 0.78rem; line-height: 1.35;
				background-color: #111827; color: #E5E7EB; padding: 1rem;
				border-radius: 0.25rem; border: 1px solid #374151;">
			{log_text}
		</div>
		""",
		unsafe_allow_html=True,
	)


def evaluate_path(agent, environment, algorithm):
	state = environment.start_state
	path = [state]
	total_reward = 0.0
	max_steps = environment.grid_size * environment.grid_size * 2

	for _ in range(max_steps):
		if algorithm == "EQLBPW":
			action = agent.e_greedy(state)
		else:
			action = agent.epsilon_greedy(state)
		next_state, reward, terminal = environment.take_step(state, action)
		total_reward += reward
		if next_state == state and not terminal:
			break
		path.append(next_state)
		state = next_state
		if terminal:
			break
	return path, total_reward, state == environment.end_state


def build_environment(environment_class, agent, preset, episodes, dynamic,
					  dynamic_count):
	environment = environment_class(
		grid=preset["grid_size"],
		start_state=preset["start_state"],
		end_state=preset["end_state"],
		agent=agent,
		episodes=episodes,
		ep_tracker=1,
		no_of_obstacles=dynamic_count,
		static_obstacles=preset["obstacles"],
		is_dynamic_obs=dynamic,
	)
	environment.generate_obstacles()
	return environment


def train_eqlbpw(preset, settings, progress_bar, status_text, metrics,
				 log_placeholder):
	from EQLBPW.agent import Agent as EQLBPWAgent
	from EQLBPW.environment import Environment as EQLBPWEnvironment

	agent = EQLBPWAgent(
		alpha=settings["alpha"], gamma=settings["gamma"], beta=0.3,
		e=settings["epsilon"], e_min=settings["epsilon_min"],
		e_decay=settings["epsilon_decay"], no_of_actions=4,
		batch_size=settings["batch_size"], max_buffer=settings["buffer_size"],
		target_sync_freq=settings["target_sync"],
	)
	environment = build_environment(EQLBPWEnvironment, agent, preset,
									settings["episodes"], settings["dynamic"],
									settings["dynamic_count"])
	environment.tracker.print_live_grid(environment.agent_pos)
	refresh_tracker_log(log_placeholder, environment)
	rewards = []
	started = time.time()
	for episode in range(environment.episodes):
		episode_number = episode + 1
		environment.agent_pos = environment.start_state
		environment.tracker.steps_per_ep = 0
		episode_reward = 0.0
		terminal = False
		episode_started = time.time()
		while not terminal and environment.tracker.steps_per_ep < environment.max_steps:
			state = environment.agent_pos
			action = agent.e_greedy(state)
			next_state, reward, terminal = environment.take_step(state, action)
			agent.memory.push(state, action, reward, next_state, terminal)
			agent.update()
			environment.agent_pos = next_state
			episode_reward += reward
			environment.tracker.steps_per_ep += 1
			environment.tracker.steps += 1
			if reward < 0:
				environment.tracker.rewards -= reward
				environment.tracker.obstacle_encountered += 1
				environment.tracker.neg_rewards += reward
			elif reward > 0:
				environment.tracker.rewards += reward
				environment.tracker.rewards_per_ep += reward
				environment.tracker.pos_rewards += reward
				environment.tracker.goal_count += 1
		agent.decay_e()
		if episode % agent.target_sync_freq == 0:
			agent.sync_target()
		rewards.append(episode_reward)
		if episode_number % environment.ep_tracker == 0:
			environment.tracker.print_episode_summary(
				curr_ep=episode_number, max_ep=environment.episodes,
				ep_tracker=environment.ep_tracker,
				elapsed=time.time() - episode_started,
				max_steps=environment.max_steps, epsilon=agent.e,
			)
		refresh_tracker_log(log_placeholder, environment)
		progress_bar.progress((episode + 1) / environment.episodes)
		status_text.write(f"Episode {episode + 1}/{environment.episodes}")
		metrics[0].metric("Episode", episode + 1)
		metrics[1].metric("Last reward", f"{episode_reward:.2f}")
		metrics[2].metric("Epsilon", f"{agent.e:.3f}")
	agent.e = 0.0
	environment.tracker.print_learned_path()
	environment.tracker.print_total_summary(start_time=started)
	refresh_tracker_log(log_placeholder, environment)
	return agent, environment, rewards, time.time() - started


def train_qlbpw(preset, settings, progress_bar, status_text, metrics,
				log_placeholder):
	agent = QLBPWAgent(
		alpha=settings["alpha"], gamma=settings["gamma"], beta=0.3,
		e=settings["epsilon"], e_min=settings["epsilon_min"],
		e_decay=settings["epsilon_decay"], no_of_states=preset["grid_size"] ** 2,
		no_of_actions=4, max_buffer=settings["buffer_size"],
		batch_size=settings["batch_size"],
	)
	environment = build_environment(QLBPWEnvironment, agent, preset,
									settings["episodes"], settings["dynamic"],
									settings["dynamic_count"])
	environment.tracker.print_live_grid(environment.agent_pos)
	refresh_tracker_log(log_placeholder, environment)
	rewards = []
	started = time.time()
	for episode in range(environment.episodes):
		episode_number = episode + 1
		environment.agent_pos = environment.start_state
		environment.tracker.steps_per_ep = 0
		episode_reward = 0.0
		terminal = False
		episode_started = time.time()
		while not terminal and environment.tracker.steps_per_ep < environment.max_steps:
			state = environment.agent_pos
			action = agent.epsilon_greedy(state)
			next_state, reward, terminal = environment.take_step(state, action)
			agent.memory.push(state, action, reward, next_state, 0.0)
			if len(agent.memory) >= agent.batch_size:
				sample = agent.adjust_lr()
				agent.update_Q(*sample, end_state=environment.end_state,
							   obstacles=environment.obstacles)
			environment.agent_pos = next_state
			episode_reward += reward
			environment.tracker.steps_per_ep += 1
			environment.tracker.steps += 1
			if reward < 0:
				environment.tracker.rewards -= reward
				environment.tracker.obstacle_encountered += 1
				environment.tracker.neg_rewards += reward
			elif reward > 0:
				environment.tracker.rewards += reward
				environment.tracker.rewards_per_ep += reward
				environment.tracker.pos_rewards += reward
				environment.tracker.goal_count += 1
		agent.e = max(agent.e_min, agent.e * agent.e_decay)
		rewards.append(episode_reward)
		if episode_number % environment.ep_tracker == 0:
			environment.tracker.print_episode_summary(
				curr_ep=episode_number, max_ep=environment.episodes,
				ep_tracker=environment.ep_tracker,
				elapsed=time.time() - episode_started,
				max_steps=environment.max_steps, epsilon=agent.e,
			)
		refresh_tracker_log(log_placeholder, environment)
		progress_bar.progress((episode + 1) / environment.episodes)
		status_text.write(f"Episode {episode + 1}/{environment.episodes}")
		metrics[0].metric("Episode", episode + 1)
		metrics[1].metric("Last reward", f"{episode_reward:.2f}")
		metrics[2].metric("Epsilon", f"{agent.e:.3f}")
	environment.tracker.print_optimal_path()
	environment.tracker.print_total_summary(start_time=started)
	refresh_tracker_log(log_placeholder, environment)
	return agent, environment, rewards, time.time() - started


def algorithm_page(algorithm):
	st.title(f"{algorithm} Gridworld Dashboard")
	st.caption("Train an agent, inspect the learned route, and compare the map with its text representation.")

	preset_name = st.selectbox("Environment preset", [preset["name"] for preset in PRESET_ENVIRONMENTS])
	preset = next(item for item in PRESET_ENVIRONMENTS if item["name"] == preset_name)
	left_config, right_config = st.columns(2)
	with left_config:
		episodes = st.slider("Episodes", 10, 500, 100, step=10)
		alpha = st.slider("Learning rate", 0.001, 0.5, 0.1, step=0.005)
		gamma = st.slider("Discount factor", 0.5, 0.99, 0.9, step=0.01)
		epsilon = st.slider("Initial epsilon", 0.1, 1.0, 0.9, step=0.05)
	with right_config:
		epsilon_min = st.slider("Minimum epsilon", 0.01, 0.5, 0.1, step=0.01)
		epsilon_decay = st.slider("Epsilon decay", 0.90, 0.999, 0.995, step=0.001)
		batch_size = st.number_input("Batch size", 1, 256, 20)
		buffer_size = st.number_input("Replay capacity", 10, 10000, 2000)
	dynamic = st.checkbox("Dynamic obstacles")
	dynamic_count = st.slider("Dynamic obstacle count", 0, 30, 5) if dynamic else 0
	target_sync = st.number_input("Target sync frequency", 1, 100, 1) if algorithm == "EQLBPW" else 1

	settings = {
		"episodes": episodes, "alpha": alpha, "gamma": gamma,
		"epsilon": epsilon, "epsilon_min": epsilon_min,
		"epsilon_decay": epsilon_decay, "batch_size": batch_size,
		"buffer_size": buffer_size, "dynamic": dynamic,
		"dynamic_count": dynamic_count, "target_sync": target_sync,
	}
	map_column, terminal_column = st.columns(2)
	with map_column:
		st.subheader("Environment map")
		map_placeholder = st.empty()
		map_placeholder.pyplot(render_grid(
			preset["grid_size"], preset["obstacles"], preset["start_state"],
			preset["end_state"],
		))
	with terminal_column:
		st.subheader("Terminal view")
		terminal_placeholder = st.empty()
		terminal_placeholder.markdown(
			"The tracker log will appear here after training starts."
		)

	train = st.button(f"Start {algorithm} training", type="primary", use_container_width=True)
	progress_bar = st.progress(0)
	status_text = st.empty()
	metric_columns = st.columns(3)
	if train:
		trainer = train_eqlbpw if algorithm == "EQLBPW" else train_qlbpw
		agent, environment, rewards, elapsed = trainer(
			preset, settings, progress_bar, status_text, metric_columns,
			terminal_placeholder,
		)
		path, total_reward, reached_goal = evaluate_path(agent, environment, algorithm)
		map_placeholder.pyplot(render_grid(
			preset["grid_size"], environment.obstacles, preset["start_state"],
			preset["end_state"], path,
		))
		refresh_tracker_log(terminal_placeholder, environment)
		st.success(f"Training completed in {elapsed:.2f}s")
		st.subheader("Reward history")
		st.line_chart(rewards)


algorithm = st.sidebar.radio("Algorithm", ["EQLBPW", "QLBPW"])
algorithm_page(algorithm)
