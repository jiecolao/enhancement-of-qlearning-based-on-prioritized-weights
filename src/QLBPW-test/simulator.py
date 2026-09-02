from environment import Environment
from agent import Agent
from env_settings import OBSTACLES
from utility import *
import time
import tracemalloc

def simulate():
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

    env = Environment(
        grid=9,
        start_state=(0, 0),
        end_state=(8, 8),
        agent=agent,
        episodes=10,
        no_of_obstacles=0,
        static_obstacles= OBSTACLES[0]["obstacles"],
        is_dynamic_obs=True
    )

    ep_tracker = 1
    env.generate_obstacles()
    env.tracker.print_live_grid(env.agent_pos)

    for ep in range(env.episodes):
        env.agent_pos = env.start_state
        is_terminal = False

        if ep % ep_tracker == 0:
            episode_start_time = time.time()

        while not is_terminal and env.tracker.steps_per_ep <= env.max_steps:
            action = agent.epsilon_greedy(env.agent_pos)
            next_state, reward, is_terminal = env.take_step(env.agent_pos, action)

            agent.memory.push(env.agent_pos, action, reward, next_state, is_terminal)

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

            # Trackers
            env.tracker.steps_per_ep += 1
            env.tracker.steps += 1
            if reward < 0:
                env.tracker.rewards -= reward
                env.tracker.obstacle_encountered += 1
                env.tracker.neg_rewards += reward
            elif reward > 0:
                env.tracker.rewards += reward
                env.tracker.rewards_per_ep += reward
                env.tracker.pos_rewards += reward
                env.tracker.goal_count += 1

        if ep % ep_tracker == 0:
            elapsed = time.time() - episode_start_time
            env.tracker.print_episode_summary(
                curr_ep=ep, 
                max_ep=env.episodes, 
                ep_tracker=ep_tracker,
                elapsed=elapsed,
                max_steps=env.max_steps,
                epsilon=agent.e
            )

    return agent, env


if __name__ == "__main__":
    print("\n=== QLBPW Simulation ===\n\n")
    tracemalloc.start()
    start_time = time.time()

    trained_agent, trained_env = simulate()

    tracemalloc.stop()

    # Visualizers
    # visualize_learned_path_dqn(self=trained_env, agent=trained_agent)