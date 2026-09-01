from environment import Environment
from agent import Agent
import time
import tracemalloc

def print_memory_stats(label):
    current, peak = tracemalloc.get_traced_memory()
    print(f"{label} | Current: {current / (1024 * 1024):.2f} MB | Peak: {peak / (1024 * 1024):.2f} MB")


def simulate():
    BASE_OBSTACLES = {
                (1, 0),                 (4, 0),                             (8, 0),
                                                        (6, 1),
        (0, 2),                 (3, 2),
                        (2, 3),                 (5, 3),         (7, 3),     (8, 3), 
        (0, 4),                 (3, 4),
                                                (5, 5), (6, 5), (7, 5), 
                (1, 6),                         (5, 6),         (7, 6), 
                                (3, 7),         (5, 7),         (7, 7),
        (0, 8)
    }

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
        max_buffer=1000,
    )

    env = Environment(
        grid = 15,
        start_state = (0, 0),
        end_state = (14, 14),
        agent = agent,
        episodes = 50,
        no_of_obstacles = 5,
        static_obstacles = BASE_OBSTACLES,
        is_dynamic_obs = False,
    )

    target_sync_freq = 10
    start_time = time.time() 

    for ep in range(env.episodes):
        curr_state = env.start_state
        episode_reward = 0.0
        is_terminal = False

        while not is_terminal and env.tracker.steps_per_ep <= env.max_steps:
            action = agent._e_greedy(curr_state)
            next_state, reward, is_terminal = env._take_step(curr_state, action)

            agent.memory.push(curr_state, action, reward, next_state, is_terminal)
            agent._update()

            curr_state = next_state
            episode_reward += reward

            env.tracker.steps_per_ep += 1

        if ep % target_sync_freq == 0:
            agent._sync_target()

        if ep % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Episode {ep:3d} | Total Steps: {env.tracker.steps_per_ep:3d} | Total Reward: {episode_reward:6.2f} | Epsilon: {agent.e:.3f} | Elapsed: {elapsed:.2f}")
            env.tracker.steps_per_ep = 0

        agent._decay_e()

if __name__ == "__main__":
    tracemalloc.start()
    start_time = time.time()
    simulate()
    elapsed_time = time.time() - start_time
    print_memory_stats("\nDQN final memory")
    print(f"DQN total runtime: {elapsed_time:.2f} seconds")
    tracemalloc.stop()