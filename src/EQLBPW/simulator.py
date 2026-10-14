from .environment import Environment
from .agent import Agent
from ..visualizer import Visualizer
from dataclasses import asdict
import time
import tracemalloc

def simulate(agent: Agent, env: Environment):
    tracemalloc.start()
    start_time = time.time()

    env.generate_obstacles()                            # Initialize obstacles
    env.tracker.print_live_grid(env.agent_pos)          # Display Grid in Terminal 
    interval_start_time = time.time()

    for ep in range(env.episodes):
        state = env.reset()
        env.tracker.steps_per_ep = 0
        env.tracker.rewards_per_ep = 0
        is_terminal = False

        episode_number = ep + 1

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

        training_progress = ep / max(env.episodes - 1, 1)
        agent.update_beta(training_progress)

        agent.decay_e() 

        if episode_number % agent.target_sync_freq == 0:
            agent.sync_target()

        # Tracker
        env.tracker.record_episode(success = is_terminal and env.agent_pos == env.end_state)

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
    from ..agent_config import EQLBPWAgent

    agent = Agent(**asdict(EQLBPWAgent()))
    env = Environment(agent=agent, **asdict(Intramuros()))

    print("\n" + "="*40)
    print("EQLBPW Simulation")
    print("="*40)

    tracemalloc.start()
    start_time = time.time()

    trained_agent, trained_env = simulate(agent=agent, env=env)

    # Visuals
    visual = Visualizer(agent=trained_agent, env=trained_env)
    visual.eqlbpqw_visualize_learned_path(save_fig=True)

    tracemalloc.stop()