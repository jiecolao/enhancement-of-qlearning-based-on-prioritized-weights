from .environment import Environment
from .agent import Agent
from ..visualizer import Visualizer
from dataclasses import asdict
import tracemalloc
import numpy as np
import time

def simulate(agent: Agent, env: Environment):
    # Original algorithm II: run the episode loop and connect environment
    # transitions, epsilon-greedy actions, replay, and Q-table updates.
    start_time = time.time()

    env.generate_obstacles()
    env.tracker.print_live_grid(env.agent_pos)
    interval_start_time = time.time()

    for ep in range(env.episodes):
        # Original II.1: initialize state s at the configured start position.
        episode_number = ep + 1
        env.agent_pos = env.start_state
        env.steps = 0
        env.tracker.steps_per_ep = 0
        env.tracker.rewards_per_ep = 0
        is_terminal = False

        while not is_terminal and env.tracker.steps_per_ep < env.max_steps:
            # Original II.a: choose a using epsilon-greedy policy.
            action = agent.epsilon_greedy(env.agent_pos)
            # Original II.b: execute a and observe r, s', and termination.
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
            # Original II.b-c: store the transition and its current TD error;
            # the latter determines its rank during priority sampling.
            agent.memory.push(env.agent_pos, action, reward, next_state, td_error)

            if len(agent.memory) > 0:
                # Original II.c-e: rank-sample one stored transition, compute
                # alpha_j, and update its tabular Q-value.
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

        if episode_number % env.ep_tracker == 0:
            elapsed = time.time() - interval_start_time
            env.tracker.epsilon_history.append(agent.e)
            env.tracker.ep_history.append(episode_number)
            env.tracker.print_episode_summary(
            curr_ep=episode_number, 
                max_ep=env.episodes, 
                ep_tracker=env.ep_tracker,
                elapsed=elapsed,
                max_steps=env.max_steps,
                epsilon=agent.e
            )
            interval_start_time = time.time()
            env.tracker.print_learned_path()    # Tracker

        if env.is_dynamic_obs and episode_number % 10 == 0:
            env.generate_obstacles()            # Dynamic Obstacle

    env.tracker.print_total_summary(start_time=start_time)
    agent.save(agent_name="test", save_memory=True)

    return agent, env


if __name__ == "__main__":
    from ..env_config import Intramuros
    from ..agent_config import QLBPWAgent

    agent = Agent(**asdict(QLBPWAgent()))
    env = Environment(agent=agent, **asdict(Intramuros()))
        
    print("\n" + "="*40)
    print("QLBPW Simulation")
    print("="*40)

    tracemalloc.start()
    start_time = time.time()

    trained_agent, trained_env = simulate(agent=agent, env=env)


    # Visuals
    visual = Visualizer(agent=trained_agent, env=trained_env)
    visual.qlbpw_visualize_learned_path(agent=trained_agent, env=trained_env, save_fig=True)

    tracemalloc.stop()