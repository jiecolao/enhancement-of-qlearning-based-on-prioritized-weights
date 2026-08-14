import numpy as np
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def generate_base_obstacles(grid_size, num_obstacles, start_state, goal_state, seed=None, blocked=None):
    rng = random.Random(seed)
    blocked_states = set(blocked or [])
    blocked_states.update([start_state, goal_state])

    all_states = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    candidates = [state for state in all_states if state not in blocked_states]

    if num_obstacles > len(candidates):
        raise ValueError("num_obstacles exceeds available free cells")

    return set(rng.sample(candidates, num_obstacles))

class QLBPW():
    def __init__(self, environment, episodes, alpha, gamma, epsilon, beta, dynamic_obs, num_dynamic_obs=5):
        self.episodes = episodes
        self.initial_alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta

        self.dynamic_obs_enabled = dynamic_obs
        self.dynamic_obs_enabled = dynamic_obs
        self.num_dynamic_obs = num_dynamic_obs

        self.grid_rows = environment['grid']
        self.grid_cols = environment['grid']
        actions = ["up", "right", "down", "left"]
        self.no_of_actions = len(actions)
        
        self.start_state = environment['start']
        self.goal_state = environment['goal']
        self.static_obstacles = environment['base_obstacles']

        self.obstacles = list(environment['base_obstacles'])

        self.buffer = []
        self.maxcap = 5000
        self.pos = 0

        self.goalCount = 0
        self.obstaclesCount = 0

    def generate_dynamic_obstacles(self):
        self.obstacles.clear()
        self.obstacles = list(self.static_obstacles)

        if not self.dynamic_obs_enabled:
            return

        dynamic_added = 0
        while dynamic_added < self.num_dynamic_obs:
            rand_x = random.randint(0, self.grid_cols - 1)
            rand_y = random.randint(0, self.grid_rows - 1)
            rand_state = (rand_x, rand_y)
            
            if (rand_state != self.start_state and 
                rand_state != self.goal_state and 
                rand_state not in self.obstacles):
                
                self.obstacles.append(rand_state)
                dynamic_added += 1

    def epsilon_greedy(self, Q, state):
        a = random.random()
        if a < self.epsilon:
            return random.randrange(self.no_of_actions)
        else:
            q_values = Q.get(state, np.zeros(self.no_of_actions))
            return np.argmax(q_values)

    def adjust_learning_rate(self):
        b = len(self.buffer)

        errors = np.array([abs(exp[4]) for exp in self.buffer])
        
        sorted_indices = np.argsort(-errors)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, b + 1)

        p_j_unnormalized = 1.0 / ranks
        p_j = p_j_unnormalized / np.sum(p_j_unnormalized)
        
        sampled_idx = np.random.choice(b, p=p_j)
        state, action, reward, next_state, td_error = self.buffer[sampled_idx]
        
        p_sampled = p_j[sampled_idx]
        adjusted_lr = self.initial_alpha / ((b * p_sampled) ** self.beta) 

        return state, action, reward, next_state, td_error, sampled_idx, adjusted_lr

    def er_add_experience(self, state, action, reward, next_state, td_error):
        experience = [state, int(action), float(reward), next_state, float(td_error)]
        
        if len(self.buffer) < self.maxcap:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.pos = (self.pos + 1) % self.maxcap

    def er_update(self, Q, state, action, reward, next_state, td_error, sampled_idx, adjusted_lr):
        if not self.buffer:
            return Q

        if state not in Q:
            Q[state] = np.zeros(self.no_of_actions)
        
        current_q = Q[state][action]
        
        if next_state == self.goal_state or next_state in self.obstacles:
            td_target = reward
        else:
            if next_state not in Q:
                Q[next_state] = np.zeros(self.no_of_actions)
            max_q_next = np.max(Q[next_state])
            td_target = reward + self.gamma * max_q_next
        
        new_td_error = td_target - current_q 
        
        Q[state][action] = (1 - adjusted_lr) * current_q + (adjusted_lr * td_target)
        
        self.buffer[sampled_idx][4] = float(new_td_error)
        
        return Q
    
    def take_step(self, state, action):
        x, y = state

        if action == 0:
            y = max(0, y - 1)
        elif action == 1:
            x = min(self.grid_cols - 1, x + 1)
        elif action == 2:
            y = min(self.grid_rows - 1, y + 1)
        elif action == 3:
            x = max(0, x - 1)

        next_state = (x, y)

        is_terminal = False
        
        if next_state in self.obstacles:
            reward = -1
            self.obstaclesCount += 1
            next_state = state
        elif next_state == self.goal_state:
            reward = 1
            is_terminal = True
            self.goalCount += 1
        else:
            reward = 0

        return next_state, reward, is_terminal

    def print_actions(self, Q):
        print("\n" + "="*40)
        print("LEARNED POLICY (Best Actions)")
        print("="*40)
        action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        
        for y in range(self.grid_rows):
            row_str = ""
            for x in range(self.grid_cols):
                state = (x, y)
                
                if state == self.goal_state:
                    row_str += " 🏁 \t"
                elif state == self.start_state:
                    row_str += " 🤖 \t"
                elif state in self.obstacles:
                    row_str += " 🧱 \t"
                else:
                    if state not in Q or np.max(Q[state]) == 0:
                        row_str += " . \t" 
                    else:
                        best_action = np.argmax(Q[state])
                        row_str += f" {action_symbols[best_action]} \t"
            print(row_str)
        print("="*40)
        

    def print_q_table(self, Q):
        print("\n" + "="*40)
        print("MAX Q-VALUES")
        print("="*40)
        for y in range(self.grid_rows):
            row_str = ""
            for x in range(self.grid_cols):
                state = (x, y)
                
                if state == self.goal_state:
                    row_str += " 🏁 \t"
                elif state == self.start_state:
                    row_str += " 🤖 \t"
                elif state in self.obstacles:
                    row_str += " 🧱 \t"
                else:
                    if state not in Q:
                        row_str += " . \t"
                    else:
                        max_val = np.max(Q[state])
                        min_val = np.min(Q[state])
                        
                        if max_val == 0.0 and min_val < 0:
                            row_str += f"{min_val:.2f}\t"
                        else:
                            row_str += f"{max_val:.2f}\t"
            print(row_str)
        print("-" * 40)

    def print_grid(self):
        print("\n" + "="*40)
        print("ENVIRONMENT")
        print("="*40)
        for y in range(self.grid_rows):
            row_str = ""
            for x in range(self.grid_cols):
                state = (x, y)
                
                if state == self.start_state:
                    row_str += " 🤖 \t"
                elif state == self.goal_state:
                    row_str += " 🏁 \t"
                elif state in self.obstacles:
                    row_str += " 🧱 \t"
                else:
                    row_str += " . \t"
            print(row_str)
        print("="*40)

    def print_agent_loc(self, curr_state):
        print("\n" + "="*40)
        print("AGENT LOCATION")
        print("="*40)
        for y in range(self.grid_rows):
            row_str = ""
            for x in range(self.grid_cols):
                state = (x, y)
                
                if state == curr_state:
                    row_str += " 🤖 \t"
                elif state == self.start_state:
                    row_str += " S \t"
                elif state == self.goal_state:
                    row_str += " 🏁 \t"
                elif state in self.obstacles:
                    row_str += " 🧱 \t"
                else:
                    row_str += " . \t"
            print(row_str)
        print("="*40)
        time.sleep(0.5)

    def print_optimal_path(self, Q):
        print("\n" + "="*40)
        print("OPTIMAL PATH")
        print("="*40)
        
        curr_state = self.start_state
        path = [curr_state]
        is_terminal = False
        steps = 0
        max_steps = (self.grid_rows * self.grid_cols) * 2

        while not is_terminal and steps < max_steps:
            if curr_state not in Q:
                break
            best_action = np.argmax(Q[curr_state])
            next_state, _, is_terminal = self.take_step(curr_state, best_action)
            path.append(next_state)
            curr_state = next_state
            steps += 1

        if curr_state != self.goal_state:
            print("<!> Warning: Agent got stuck and didn't reach the goal.")

        for y in range(self.grid_rows):
            row_str = ""
            for x in range(self.grid_cols):
                state = (x, y)
                
                if state == self.start_state:
                    row_str += " 🤖 \t"
                elif state == self.goal_state:
                    row_str += " 🏁 \t"
                elif state in self.obstacles:
                    row_str += " 🧱 \t"
                elif state in path:
                    row_str += " 🟢 \t"
                else:
                    row_str += " . \t"
            print(row_str)
            
        print(f"\nSteps taken: {len(path) - 1}")
        print("="*40)

    def visualize_learned_path(self, Q, title="Q-Learning Optimal Path"):
        """Visualize the optimal path learned by Q-Learning using Matplotlib"""
        curr_state = self.start_state
        path = [curr_state]
        is_terminal = False
        steps = 0
        max_steps = (self.grid_rows * self.grid_cols) * 2

        while not is_terminal and steps < max_steps:
            if curr_state not in Q:
                break
            best_action = np.argmax(Q[curr_state])
            next_state, _, is_terminal = self.take_step(curr_state, best_action)
            path.append(next_state)
            curr_state = next_state
            steps += 1

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.set_xlim(-0.5, self.grid_cols - 0.5)
        ax.set_ylim(self.grid_rows - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        for obs in self.obstacles:
            rect = patches.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1,
                                    linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)

        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'g-', linewidth=2, alpha=0.6, label='Learned Path')
            ax.scatter(path_x, path_y, c='green', s=20, alpha=0.5)

        ax.scatter(*self.goal_state, c='red', s=300, marker='*', label='Goal', zorder=5)
        ax.scatter(*self.start_state, c='blue', s=200, marker='o', label='Start', zorder=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()

        print(f"Path length: {len(path) - 1} steps")
        plt.show()

    def plot_learning_curves(self, steps, rewards):
        print("Generating learning curves...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(steps, color='blue', alpha=0.7, linewidth=1)
        ax1.set_title("Convergence: Episode via Steps")
        ax1.set_xlabel("Episodes")
        ax1.set_ylabel("Steps to Reach Goal / Terminate")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        ax2.plot(rewards, color='green', alpha=0.7, linewidth=1)
        ax2.set_title("Convergence: Episode via Reward")
        ax2.set_xlabel("Episodes")
        ax2.set_ylabel("Total Episode Reward")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.show()

    def simulate_qlbpw(self, start_time):

        Q = {}

        optimal_path_length = 16
        optimal_time_recorded = False   
        expected_time = 27
        track_time = True
        e_tracker = 100

        steps_per_episode = []
        rewards_per_episode = []

        self.print_grid()

        for e in range(self.episodes):
            if e % 100 == 0 and self.dynamic_obs_enabled: self.generate_dynamic_obstacles()               
            # if e == 699 and self.dynamic_obs_enabled: self.obstacles.append((1, 4))
            # self.generate_dynamic_obstacles()
            curr_state = self.start_state

            # self.gamma = 0.1 + (0.9 - 0.1) * (e / max(1, self.episodes - 1))
            self.epsilon = 0.9 - (0.9 - 0.1) * (e / max(1, self.episodes - 1))
            is_terminal = False
            steps_taken = 0
            max_step = self.grid_cols * self.grid_cols ** 2
            episode_reward = 0

            # while not is_terminal:
            while not is_terminal and steps_taken <= max_step:
                action = self.epsilon_greedy(Q, curr_state)

                next_state, reward, is_terminal = self.take_step(curr_state, action)

                steps_taken += 1 
                episode_reward += reward
                if ((time.time() - start_time) >= float(expected_time)) and optimal_time_recorded == False and track_time:
                    track_time = False

                if curr_state not in Q:
                    Q[curr_state] = np.zeros(self.no_of_actions)
                
                current_q = Q[curr_state][action]
                
                if is_terminal:
                    td_target = reward
                else:
                    if next_state not in Q:
                        Q[next_state] = np.zeros(self.no_of_actions)
                    max_q_next = np.max(Q[next_state])
                    td_target = reward + self.gamma * max_q_next

                td_error = td_target - current_q

                self.er_add_experience(curr_state, action, reward, next_state, td_error)

                if len(self.buffer) > 0:
                    (sampled_state, sampled_action, sampled_reward, 
                     sampled_next_state, sampled_td_error, 
                     sampled_idx, adjusted_lr) = self.adjust_learning_rate()
                    Q = self.er_update(Q, sampled_state, sampled_action, sampled_reward, 
                                       sampled_next_state, sampled_td_error, 
                                       sampled_idx, adjusted_lr)

                curr_state = next_state

            steps_per_episode.append(steps_taken)
            rewards_per_episode.append(episode_reward)

            if curr_state == self.goal_state and steps_taken == optimal_path_length:
                if not optimal_time_recorded:
                    time_to_optimal = time.time() - start_time
                    optimal_time_recorded = True
            
            if (e + 1) % e_tracker == 0:
                elapsed = time.time() - start_time
                self.print_q_table(Q)
                print(f"Goal Found: {self.goalCount}")
                print(f"Obstacles Encountered: {self.obstaclesCount}")
                print(f"Episode {e + 1}/{self.episodes} | Elapsed: {elapsed:.2f}s | Steps: {steps_taken}")
                self.goalCount = 0
                self.obstaclesCount = 0

        self.print_optimal_path(Q)
        self.visualize_learned_path(Q, title=f"{self.grid_rows}x{self.grid_cols} Q-Learning Optimal Path | Elapsed: {time.time() - start_time:.2f}s")
        print(f"Total Episodes: {self.episodes}")

if __name__ == "__main__":
    BASE_OBSTACLES = {
        (1, 0), (4, 0), (8, 0),
        (6, 1),
        (0, 2), (3, 2),
        (2, 3), (5, 3), (7, 3), (8, 3), 
        (0, 4), (3, 4),
        (6, 5), (7, 5), (5, 5), 
        (1, 6), (5, 6), (7, 6), 
        (3, 7), (5, 7), (7, 7),
        (0, 8)
    }

    environment = [
        {
            'name': '9x9',
            'grid': 9,
            'start': (0, 0),
            'goal': (6, 6),
            'base_obstacles': BASE_OBSTACLES,
        },
        {
            'name': '10x10',
            'grid': 10,
            'start': (0, 0),
            'goal': (9, 9),
            'base_obstacles': generate_base_obstacles(
                grid_size=10,
                num_obstacles=20,
                start_state=(0, 0),
                goal_state=(9, 9),
                seed=None
            )
        },
        {
            'name': '15x15',
            'grid': 15,
            'start': (0, 0),
            'goal': (14, 14),
            'base_obstacles': generate_base_obstacles(
                grid_size=15,
                num_obstacles=30,
                start_state=(0, 0),
                goal_state=(14, 14),
                seed=None
            )
        },
        {
            'name': '20x20',
            'grid': 20,
            'start': (0, 0),
            'goal': (19, 19),
            'base_obstacles': generate_base_obstacles(
                grid_size=10,
                num_obstacles=40,
                start_state=(0, 0),
                goal_state=(19, 19),
                seed=None
            )
        },
    ]

    a = QLBPW(
        environment=environment[0],
        episodes=1000, 
        alpha=0.1, 
        gamma=0.9, 
        epsilon=0.9, 
        beta=0.3,
        dynamic_obs=True,
        num_dynamic_obs=1
    )

    b = QLBPW(
        environment=environment[1],
        episodes=1000, 
        alpha=0.1, 
        gamma=0.9, 
        epsilon=0.9, 
        beta=0.3,
        dynamic_obs=True,
        num_dynamic_obs=1
    )

    c = QLBPW(
        environment=environment[2],
        episodes=1000, 
        alpha=0.1, 
        gamma=0.9, 
        epsilon=0.9, 
        beta=0.3,
        dynamic_obs=True,
        num_dynamic_obs=1
    )

    d = QLBPW(
        environment=environment[3],
        episodes=1000, 
        alpha=0.1, 
        gamma=0.9, 
        epsilon=0.9, 
        beta=0.3,
        dynamic_obs=True,
        num_dynamic_obs=1
    )

    # simulations = [a, b, c, d]
    # index = 0
    # for i in simulations:
    #     print(f"\nStarting {environment[index]["name"]} simulation...")
    #     start_time = time.time() # Start stopwatch
    #     i.simulate_qlbpw(start_time) 
    #     end_time = time.time() # Stop stopwatch
    #     elapsed_time = end_time - start_time
    #     index += 1

    # print(f"\nStarting simulation...")
    # start_time = time.time() 
    # QLBPW(
    #     environment=environment[0],
    #     episodes=1000, 
    #     alpha=0.1, 
    #     gamma=0.9, 
    #     epsilon=0.9, 
    #     beta=0.3,
    #     dynamic_obs=True,
    #     num_dynamic_obs=1
    # ).simulate_qlbpw(start_time)

    # end_time = time.time() 
    # elapsed_time = end_time - start_time