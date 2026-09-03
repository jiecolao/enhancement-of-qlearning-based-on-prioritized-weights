from environment import Environment
from agent import Agent
import time
from env_settings import OBSTACLES
from utility import *
from visualizer import Visualizer

def simulate():
    agent = Agent(
        alpha=0.1,
        gamma=0.9,
        beta=0.3,
        e=0.9, 
        e_min=0.1, 
        e_decay=0.999,        
        no_of_states=4, 
        no_of_actions=4,
        batch_size=20, 
        max_buffer=2000,
    )

    env = Environment(
        grid = 20,
        start_state = (4, 0),
        end_state = (15, 13),
        agent = agent,
        episodes = 10,
        no_of_obstacles = 10,
        static_obstacles = OBSTACLES[1]["obstacles"],
        is_dynamic_obs = True,
    )

    target_sync_freq = 10
    ep_tracker = 5

    env.generate_obstacles()
    # env.tracker.print_live_grid(env.agent_pos)

    for ep in range(env.episodes):
        env.agent_pos = env.start_state
        episode_reward = 0.0
        is_terminal = False
        
        if ep % ep_tracker == 0:
            episode_start_time = time.time()

        while not is_terminal and env.tracker.steps_per_ep <= env.max_steps:
            action = agent.e_greedy(env.agent_pos)
            next_state, reward, is_terminal = env.take_step(env.agent_pos, action)

            agent.memory.push(env.agent_pos, action, reward, next_state, is_terminal)
            agent.update()

            env.agent_pos = next_state
            episode_reward += reward

            # Trackers
            env.tracker.steps_per_ep += 1
            env.tracker.steps += 1
            if reward < 0:
                env.tracker.rewards -= 1
                env.tracker.obstacle_encountered += 1
                env.tracker.neg_rewards += 1
            elif reward > 0:
                env.tracker.rewards += 1
                env.tracker.rewards_per_ep += reward
                env.tracker.pos_rewards += 1
                env.tracker.goal_count += 1

        if ep % target_sync_freq == 0:
            agent.sync_target()

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
            
        agent.decay_e() 

        if ep == env.episodes-1:     
            elapsed_time = time.time() - start_time
            print(f"EQLBPW Total Runtime: {elapsed_time:.2f} seconds")
            print(f"EQLBPW Total Rewards: {env.tracker.rewards}")

    return agent, env

if __name__ == "__main__":
    print("\n=== EQLBPW-1 Simulation ===\n\n")

    tracemalloc.start()
    start_time = time.time()

    trained_agent, trained_env = simulate()

    # Visualizer
    visual = Visualizer(
        agent=trained_agent,
        env=trained_env
    )
    visual.dqn_visualize_learned_path()
    
    tracemalloc.stop()