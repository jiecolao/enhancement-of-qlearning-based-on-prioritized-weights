from .environment import Environment
from .agent import Agent
from env_settings import OBSTACLES
from visualizer import Visualizer
import time
import tracemalloc

def simulate():
    state_dim = 29
    learning_rate = 0.0005
    gamma = 0.95
    priority_alpha = 0.6
    beta_start = 0.4
    beta_end = 1.0
    e = 1.0
    e_min = 0.05
    e_decay = 0.995
    no_of_actions = 4
    batch_size = 64
    max_buffer = 50000
    target_sync_freq = 20

    collision_weight = 1.0
    goal_weight = 2.0
    distance_weight = 0.5

    agent = Agent(
        state_dim=state_dim,
        action_dim=no_of_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        priority_alpha=priority_alpha,
        beta_start=beta_start,
        beta_end=beta_end,
        e=e,
        e_min=e_min,
        e_decay=e_decay,
        max_buffer=max_buffer,
        batch_size=batch_size,
        target_sync_freq=target_sync_freq,
        collision_weight=collision_weight,
        goal_weight=goal_weight,
        distance_weight=distance_weight,
    )

    grid = 20
    start = (4, 0)
    end = (9, 19)
    episodes = 2000
    ep_tracker = 10
    no_of_obstacles = 0
    static_obstacles = OBSTACLES[1]["obstacles"]
    is_dynamic_obs = False

    env = Environment(
        grid=grid,
        start_state=start,
        end_state=end,
        agent=agent,
        episodes=episodes,
        ep_tracker=ep_tracker,
        no_of_obstacles=no_of_obstacles,
        static_obstacles=static_obstacles,
        is_dynamic_obs=is_dynamic_obs,
    )

    env.generate_obstacles()                            # Initialize obstacles
    env.tracker.print_live_grid(env.agent_pos)          # Display Grid in Terminal 

    for ep in range(env.episodes):
        state = env.reset()
        is_terminal = False

        episode_number = ep + 1

        # Tracker
        if episode_number % env.ep_tracker == 0:
            episode_start_time = time.time()

        while not is_terminal and env.tracker.steps_per_ep < env.max_steps:
            action = agent.e_greedy(state)

            next_position, reward, is_terminal, info = env.take_step(env.agent_pos, action)

            next_state = env.get_state()

            agent.memory.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=is_terminal,
                collision=info["collision"],
                goal=info["goal"],
                distance_progress=info["distance_progress"],
            )

            agent.update()

            state = next_state

            # Trackers
            env.tracker.steps_per_ep += 1
            env.tracker.steps += 1
            env.tracker.rewards += reward
            env.tracker.rewards_per_ep += reward

            if reward < 0:
                env.tracker.neg_rewards += reward

                if info["collision"]:
                    env.tracker.obstacle_encountered += 1

            elif reward > 0:
                env.tracker.pos_rewards += reward

                if info["goal"]:
                    env.tracker.goal_count += 1

        training_progress = ep / max(episodes - 1, 1)
        agent.update_beta(training_progress)

        agent.decay_e() 

        if episode_number % agent.target_sync_freq == 0:
            agent.sync_target()

        # Tracker

        if episode_number % env.ep_tracker == 0:
            elapsed = time.time() - episode_start_time

            env.tracker.record_episode(success = is_terminal and env.agent_pos == env.end_state)

            env.tracker.print_episode_summary(
                curr_ep=episode_number,
                max_ep=env.episodes,
                ep_tracker=env.ep_tracker,
                elapsed=elapsed,
                max_steps=env.max_steps,
                epsilon=agent.e
            )
        else: 
            env.tracker.record_episode(success=is_terminal and env.agent_pos == env.end_state)

        if episode_number % 100 == 0:
            env.tracker.print_learned_path()

    return agent, env

if __name__ == "__main__":
    print("\n" + "="*40)
    print("EQLBPW Simulation")
    print("="*40)

    tracemalloc.start()
    start_time = time.time()
    trained_agent, trained_env = simulate()

    # trained_env.tracker.print_optimal_path()
    trained_env.tracker.print_total_summary(start_time=start_time)
    # trained_agent.save(agent_name="test", save_memory=True)

    # Visuals
    visual = Visualizer(agent=trained_agent, env=trained_env)
    visual.eqlbpqw_visualize_learned_path(save_fig=True)

    tracemalloc.stop()