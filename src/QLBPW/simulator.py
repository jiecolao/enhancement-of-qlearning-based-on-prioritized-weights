from .environment import Environment
from .agent import Agent
from env_settings import PRESET_ENVIRONMENTS
from visualizer import Visualizer
import tracemalloc
import numpy as np
import time

def simulate(grid_size=None, episodes=100, preset=None):

    agent = Agent(
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
    
    if preset is None:
        preset = next(
            environment for environment in PRESET_ENVIRONMENTS
            if (
                environment["grid_size"] == 20
                and isinstance(environment["start_state"], tuple)
                and isinstance(environment["end_state"], tuple)
            )
        )

    grid = grid_size if grid_size is not None else preset["grid_size"]

    start_state = preset["start_state"]
    end_state = preset["end_state"]
    obstacles = preset["obstacles"]

    env = Environment(
        grid=grid,
        start_state=start_state,
        end_state=end_state,
        agent=agent,
        episodes=episodes,
        ep_tracker=10,
        no_of_obstacles=0,
        static_obstacles=obstacles,
        is_dynamic_obs=False
    )

    env.generate_obstacles()
    env.tracker.print_live_grid(env.agent_pos)

    for ep in range(env.episodes):
        episode_number = ep + 1
        env.agent_pos = env.start_state
        env.steps = 0
        env.tracker.steps_per_ep = 0
        env.tracker.rewards_per_ep = 0
        is_terminal = False

        if episode_number % env.ep_tracker == 0:
            episode_start_time = time.time()

        while not is_terminal and env.tracker.steps_per_ep < env.max_steps:
            action = agent.epsilon_greedy(env.agent_pos)
            next_state, reward, is_terminal = env.take_step(env.agent_pos, action)

            if env.agent_pos not in agent.Q:
                agent.Q[env.agent_pos] = np.zeros(agent.no_of_actions)

            current_q = agent.Q[env.agent_pos][action]

            if is_terminal:
                td_target = reward
            else:
                if next_state not in agent.Q:
                    agent.Q[next_state] = np.zeros(agent.no_of_actions)
                max_q_next = np.max(agent.Q[next_state])
                td_target = reward + agent.gamma * max_q_next

            td_error = td_target - current_q
            agent.memory.push(env.agent_pos, action, reward, next_state, td_error)

            if len(agent.memory) > 0:
                (sampled_state, sampled_action, sampled_reward, 
                    sampled_next_state, sampled_td_error, 
                    sampled_idx, adjusted_lr) = agent.adjust_lr()
                agent.Q = agent.update_Q(
                    state=sampled_state, 
                    action=sampled_action, 
                    reward=sampled_reward, 
                    next_state=sampled_next_state, 
                    td_error=sampled_td_error, 
                    sampled_idx=sampled_idx, 
                    adjusted_lr=adjusted_lr,
                    end_state=env.end_state,
                    obstacles=env.obstacles
                )

            env.agent_pos = next_state
            env.steps += 1

            # Trackers
            env.tracker.steps_per_ep += 1
            env.tracker.steps += 1
            env.tracker.rewards += reward
            env.tracker.rewards_per_ep += reward
            if reward < 0:
                env.tracker.obstacle_encountered += 1
                env.tracker.neg_rewards += reward
            elif reward > 0:
                env.tracker.pos_rewards += reward
                env.tracker.goal_count += 1

        env.tracker.record_episode(
            success=env.agent_pos == env.end_state
        )

        env.tracker.record_qtable_metrics()

        if episode_number % env.ep_tracker == 0:
            elapsed = time.time() - episode_start_time
            env.tracker.print_episode_summary(
            curr_ep=episode_number, 
                max_ep=env.episodes, 
                ep_tracker=env.ep_tracker,
                elapsed=elapsed,
                max_steps=env.max_steps,
                epsilon=agent.e
            )

        if episode_number % 10 == 0:
            # env.tracker.print_learned_path()    # Tracker
            env.generate_obstacles()            # Dynamic Obstacle

    return agent, env


if __name__ == "__main__":
    print("\n" + "="*40)
    print("QLBPW Simulation")
    print("="*40)
    tracemalloc.start()
    start_time = time.time()

    trained_agent, trained_env = simulate()

    trained_env.tracker.print_learned_path()
    trained_env.tracker.print_total_summary(start_time=start_time)
    # trained_agent.save(agent_name="test", save_memory=True)

    # Visuals
    visual = Visualizer(agent=trained_agent, env=trained_env)
    visual.qlbpw_visualize_learned_path(agent=trained_agent, env=trained_env, save_fig=True)

    tracemalloc.stop()