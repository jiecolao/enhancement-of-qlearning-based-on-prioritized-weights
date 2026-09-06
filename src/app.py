import html
import time

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

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


def evaluate_qlbpw_path(agent, environment):
	state = environment.start_state
	path = [state]
	total_reward = 0.0
	max_steps = environment.grid_size * environment.grid_size * 2

	for _ in range(max_steps):
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


def evaluate_eqlbpw_path(agent, environment):
	original_position = environment.agent_pos
	original_steps = environment.steps
	was_training = agent.main_net.training
	state = environment.start_state
	path = [state]
	total_reward = 0.0
	max_steps = environment.grid_size * environment.grid_size * 2

	agent.main_net.eval()
	try:
		with torch.no_grad():
			for _ in range(max_steps):
				environment.agent_pos = state
				state_features = torch.as_tensor(
					environment.get_state(), dtype=torch.float32
				).unsqueeze(0)
				action = agent.main_net(state_features).argmax(dim=1).item()
				next_state, reward, terminal, _ = environment.take_step(
					state, action
				)
				total_reward += reward
				path.append(next_state)
				state = next_state
				if terminal:
					break
	finally:
		environment.agent_pos = original_position
		environment.steps = original_steps
		agent.main_net.train(was_training)

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
	refresh_shortest_path(environment)
	return environment


def refresh_shortest_path(environment):
	tracker = getattr(environment, "tracker", None)
	if tracker is None or not hasattr(tracker, "calculate_shortest_path"):
		return
	tracker.shortest_path = tracker.calculate_shortest_path()
	tracker.shortest_path_steps = (
		len(tracker.shortest_path) - 1
		if tracker.shortest_path is not None else None
	)


def train_eqlbpw(preset, settings, progress_bar, status_text, metrics,
				 log_placeholder):
	from EQLBPW.agent import Agent as EQLBPWAgent
	from EQLBPW.environment import Environment as EQLBPWEnvironment

	agent = EQLBPWAgent(
		state_dim=29, action_dim=4,
		learning_rate=settings["alpha"], gamma=settings["gamma"],
		priority_alpha=settings["priority_alpha"],
		beta_start=settings["beta_start"], beta_end=settings["beta_end"],
		e=settings["epsilon"], e_min=settings["epsilon_min"],
		e_decay=settings["epsilon_decay"],
		batch_size=settings["batch_size"], max_buffer=settings["buffer_size"],
		target_sync_freq=settings["target_sync"],
		collision_weight=settings["collision_weight"],
		goal_weight=settings["goal_weight"],
		distance_weight=settings["distance_weight"],
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
		state = environment.reset()
		environment.tracker.steps_per_ep = 0
		environment.tracker.rewards_per_ep = 0
		episode_reward = 0.0
		terminal = False
		episode_started = time.time()
		while not terminal and environment.tracker.steps_per_ep < environment.max_steps:
			action = agent.e_greedy(state)
			next_position, reward, terminal, info = environment.take_step(
				environment.agent_pos, action
			)
			next_state = environment.get_state()
			agent.memory.push(
				state=state, action=action, reward=reward,
				next_state=next_state, done=terminal,
				collision=info["collision"], goal=info["goal"],
				distance_progress=info["distance_progress"],
			)
			agent.update()
			state = next_state
			episode_reward += reward
			environment.tracker.steps_per_ep += 1
			environment.tracker.steps += 1
			environment.tracker.rewards += reward
			environment.tracker.rewards_per_ep += reward
			if info["collision"]:
				environment.tracker.obstacle_encountered += 1
			if reward < 0:
				environment.tracker.neg_rewards += reward
			elif reward > 0:
				environment.tracker.pos_rewards += reward
			if info["goal"]:
				environment.tracker.goal_count += 1

		training_progress = episode / max(environment.episodes - 1, 1)
		agent.update_beta(training_progress)
		agent.decay_e()
		if episode_number % agent.target_sync_freq == 0:
			agent.sync_target()
		environment.tracker.record_episode(
			success=terminal and environment.agent_pos == environment.end_state
		)
		rewards.append(episode_reward)
		if settings["dynamic"] and settings["obstacle_interval"] > 0 and episode_number % settings["obstacle_interval"] == 0:
			environment.generate_obstacles()
			refresh_shortest_path(environment)
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
		if algorithm == "EQLBPW":
			priority_alpha = st.slider("Priority alpha", 0.0, 1.0, 0.6, step=0.05)
			beta_start = st.slider("Beta start", 0.0, 1.0, 0.4, step=0.05)
			beta_end = st.slider("Beta end", 0.0, 1.0, 1.0, step=0.05)
			collision_weight = st.number_input("Collision priority weight", 0.0, 10.0, 1.0, step=0.5)
			goal_weight = st.number_input("Goal priority weight", 0.0, 10.0, 2.0, step=0.5)
			distance_weight = st.number_input("Distance priority weight", 0.0, 10.0, 0.5, step=0.5)
		else:
			priority_alpha = beta_start = beta_end = 0.0
			collision_weight = goal_weight = distance_weight = 0.0
	dynamic = st.checkbox("Dynamic obstacles")
	dynamic_count = st.slider("Dynamic obstacle count", 0, 30, 5) if dynamic else 0
	obstacle_interval = st.number_input(
		"Obstacle regeneration interval", 1, 500, 100
	) if dynamic and algorithm == "EQLBPW" else 0
	target_sync = st.number_input("Target sync frequency", 1, 100, 1) if algorithm == "EQLBPW" else 1

	settings = {
		"episodes": episodes, "alpha": alpha, "gamma": gamma,
		"epsilon": epsilon, "epsilon_min": epsilon_min,
		"epsilon_decay": epsilon_decay, "batch_size": batch_size,
		"buffer_size": buffer_size, "dynamic": dynamic,
		"dynamic_count": dynamic_count, "target_sync": target_sync,
		"priority_alpha": priority_alpha, "beta_start": beta_start,
		"beta_end": beta_end, "collision_weight": collision_weight,
		"goal_weight": goal_weight, "distance_weight": distance_weight,
		"obstacle_interval": obstacle_interval,
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
		if algorithm == "EQLBPW":
			path, total_reward, reached_goal = evaluate_eqlbpw_path(
				agent, environment
			)
		else:
			path, total_reward, reached_goal = evaluate_qlbpw_path(
				agent, environment
			)
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
