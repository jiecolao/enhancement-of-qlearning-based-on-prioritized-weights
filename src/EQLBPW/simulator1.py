from environment import Environment
from agent import Agent
import time
from env_settings import OBSTACLES
from utility import *

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
        episodes = 50,
        no_of_obstacles = 10,
        static_obstacles = OBSTACLES[1]["obstacles"],
        is_dynamic_obs = True,
    )

    target_sync_freq = 10
    ep_tracker = 1

    env._generate_obstacles()
    
    env.tracker.print_live_grid(env.agent_pos)

    for ep in range(env.episodes):
        env.agent_pos = env.start_state
        episode_reward = 0.0
        is_terminal = False
        
        if ep % ep_tracker == 0:
            episode_start_time = time.time()

        while not is_terminal and env.tracker.steps_per_ep <= env.max_steps:
            action = agent._e_greedy(env.agent_pos)
            next_state, reward, is_terminal = env._take_step(env.agent_pos, action)

            agent.memory.push(env.agent_pos, action, reward, next_state, is_terminal)
            agent._update()

            env.agent_pos = next_state
            episode_reward += reward

            env.tracker.steps_per_ep += 1

            if is_terminal:
                env.tracker.rewards_per_ep += reward

            if reward < 0:
                env.tracker.obstacle_encountered += 1

        if ep % target_sync_freq == 0:
            agent._sync_target()

        if ep % ep_tracker == 0:
            elapsed = time.time() - episode_start_time
            print(f"Episode {ep:3d} | Total Steps: {env.tracker.steps_per_ep:3d} | Reward: {episode_reward:6.2f} | Epsilon: {agent.e:.3f} | Elapsed: {elapsed:.2f}")
            print_memory_stats("Memory Status")
            env.tracker.steps_per_ep = 0

        agent._decay_e() 

    return env.tracker.rewards_per_ep

if __name__ == "__main__":
    print("\n=== EQLBPW-1 Simulation ===\n\n")
    tracemalloc.start()
    start_time = time.time()
    total_rewards = simulate()
    elapsed_time = time.time() - start_time

    # print_memory_stats("DQN final memory")
    print(f"\nEQLBPW Total Runtime: {elapsed_time:.2f} seconds")
    print(f"EQLBPW Total Rewards: {total_rewards}")
    tracemalloc.stop()