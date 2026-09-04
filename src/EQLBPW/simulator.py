from environment import Environment
from agent import Agent
from env_settings import OBSTACLES
from visualizer import Visualizer
import time
import tracemalloc

def simulate():
    agent = Agent(
        alpha=0.001,              # Learning Rate
        gamma=0.95,              # Discount Factor
        beta=0.3,               # Beta
        e=1.0,                  # Epsilon
        e_min=0.1,              # Minimun Epsilon
        e_decay=0.999,          # Epsilon Decay
        no_of_states=4,         # States
        no_of_actions=4,        # Actions: 1=up, 2=right, 3=down, 4=left 
        batch_size=64,          # The Number of Experiences To Be Sampled
        max_buffer=10000,        # Max Number of Stored Experiences
        target_sync_freq=1      # When should the Target Network sync
    )

    env = Environment(
        grid = 20,                                      # Grid Environment gridxgrid
        start_state = (4, 0),                           # Agent Starting Position
        end_state = (15, 13),                           # Finish Line
        agent = agent,                                  # Agent
        episodes = 2,                                   # Episodes to train
        ep_tracker = 1,                                 # How and when should the tracker print the summary
        no_of_obstacles = 0,                            # Number of obstacles to appear. (To spawn, set is_dynamic_obs to True)
        static_obstacles = OBSTACLES[1]["obstacles"],   # Premade obstacles
        is_dynamic_obs = True,                          # Obstacle Event Trigger
    ) 

    env.generate_obstacles()                            # Initialize obstacles
    env.tracker.print_live_grid(env.agent_pos)          # Display Grid in Terminal 

    for ep in range(env.episodes):
        env.agent_pos = env.start_state 
        episode_reward = 0.0
        is_terminal = False

        # Tracker
        if ep % env.ep_tracker == 0:
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

        if ep % agent.target_sync_freq == 0:
            agent.sync_target()

        if ep % env.ep_tracker == 0:
            elapsed = time.time() - episode_start_time
            env.tracker.print_episode_summary(
                curr_ep=ep,
                max_ep=env.episodes,
                ep_tracker=env.ep_tracker,
                elapsed=elapsed,
                max_steps=env.max_steps,
                epsilon=agent.e
            )
            
        agent.decay_e() 

    return agent, env

if __name__ == "__main__":
    print("\n" + "="*40)
    print("EQLBPW Simulation")
    print("="*40)

    tracemalloc.start()
    start_time = time.time()
    trained_agent, trained_env = simulate()

    trained_env.tracker.print_optimal_path()
    trained_env.tracker.print_total_summary(start_time=start_time)
    # trained_agent.save(agent_name="test", save_memory=True)

    # Visuals
    visual = Visualizer(
        agent=trained_agent,
        env=trained_env
    )
    # visual.eqlbpqw_visualize_learned_path(save_fig=True)

    tracemalloc.stop()