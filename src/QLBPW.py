import numpy as np
import random
import time
import matplotlib.pyplot as plt

class QLBPW():
    def __init__(self, episodes, alpha, gamma, epsilon, beta, dynamic_obs, num_dynamic_obs=5):
        self.episodes = episodes
        self.initial_alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta # superparam 

        # New dynamic obstacle settings
        self.dynamic_obs_enabled = dynamic_obs
        self.num_dynamic_obs = num_dynamic_obs

        # Environment
        self.grid_rows = 9
        self.grid_cols = 9
        actions = ["up", "right", "down", "left"]
        self.no_of_actions = len(actions)
        
        # State represented as (row, col) coordinates
        self.start_state = (0, 0)
        self.goal_state = (6, 6)

        self.static_obstacles = [
            # (1, 0), (4, 0), (8, 0),
            # (1, 1),
            # (2, 0), (2, 3),
            # (3, 2), (3, 5), (3, 7), (3, 8),
            # (4, 0), (4, 3),
            # (5, 6), (5, 7), (5, 5),
            # (6, 1), (6, 5), (6, 7),
            # (7, 3), (7, 5), (7, 7),
            # (8, 0)
        ]

        self.obstacles = []

        # Prioritized Experience Replay (Empirical experience)
        self.buffer = []
        self.maxcap = 5000
        self.pos = 0

        self.goalCount = 0
        self.obstaclesCount = 0

    def generate_dynamic_obstacles(self):
        # Reset the obstacles list to just the static ones
        self.obstacles.clear()
        self.obstacles = self.static_obstacles.copy()

        if not self.dynamic_obs_enabled:
            return

        dynamic_added = 0
        while dynamic_added < self.num_dynamic_obs:
            # Pick a random coordinate on the grid
            rand_row = random.randint(0, self.grid_rows - 1)
            rand_col = random.randint(0, self.grid_cols - 1)
            rand_state = (rand_row, rand_col)
            
            # Make sure it's not the start, goal, or already an obstacle
            if (rand_state != self.start_state and 
                rand_state != self.goal_state and 
                rand_state not in self.obstacles):
                
                self.obstacles.append(rand_state)
                dynamic_added += 1

    def epsilon_greedy(self, Q, state):
        a = random.random()
        if a < self.epsilon: # exploration
            return random.randrange(self.no_of_actions)
        else: # exploitation
            # Get Q-values for this state (default to zeros if not visited)
            q_values = Q.get(state, np.zeros(self.no_of_actions))
            return np.argmax(q_values)

    def adjust_learning_rate(self):
        b = len(self.buffer)

        errors = np.array([abs(exp[4]) for exp in self.buffer])
        ranks = np.argsort(np.argsort(-errors)) + 1

        # ranks = np.arange(1, b + 1)
        p_j_unnormalized = 1.0 / ranks                    # Equation (10)
        p_j = p_j_unnormalized / np.sum(p_j_unnormalized) # Normalize to create valid probabilities
        
        # Non-uniform random sampling based on the calculated weights
        sampled_idx = np.random.choice(b, p=p_j)
        state, action, reward, next_state, td_error = self.buffer[sampled_idx]
        
        # 2. Adjust the learning rate (alpha) using Equation (11)
        p_sampled = p_j[sampled_idx]
        # a_j = alpha / (b * p_j)^beta
        adjusted_lr = self.initial_alpha / ((b * p_sampled) ** self.beta) 

        return state, action, reward, next_state, td_error, sampled_idx, adjusted_lr

    # Experience Replay
    def er_add_experience(self, state, action, reward, next_state, td_error):
        experience = [state, int(action), float(reward), next_state, float(td_error)]
        
        if len(self.buffer) < self.maxcap:
            # If the buffer isn't full yet, just append
            self.buffer.append(experience)
        else:
            # If full, overwrite the oldest memory
            self.buffer[self.pos] = experience
        
        self.pos = (self.pos + 1) % self.maxcap

    def er_update(self, Q, state, action, reward, next_state, td_error, sampled_idx, adjusted_lr):
        if not self.buffer:
            return Q

        # 3. Calculate TD Target and new TD Error (Equations 6, 7, and 8)
        # Initialize state in Q-table if not present
        if state not in Q:
            Q[state] = np.zeros(self.no_of_actions)
        
        current_q = Q[state][action]
        
        # If the next state is the end of the line, there is no future Q-value!
        if next_state == self.goal_state or next_state in self.obstacles:
            td_target = reward
        else:
            # Initialize next_state in Q-table if not present
            if next_state not in Q:
                Q[next_state] = np.zeros(self.no_of_actions)
            max_q_next = np.max(Q[next_state])
            td_target = reward + self.gamma * max_q_next
        
        # Calculate the new TD error (delta_j)
        new_td_error = td_target - current_q 
        
        # 4. Update the Q-Table (Equation 9)
        # We replace the standard alpha with our adjusted_lr (a_j)
        Q[state][action] = (1 - adjusted_lr) * current_q + (adjusted_lr * td_target)
        
        # 5. Update the error in the buffer and re-sort
        self.buffer[sampled_idx][4] = float(new_td_error)
        # self.buffer.sort(key=lambda x: abs(x[4]), reverse=True)
        
        return Q
    
    def take_step(self, state, action):
        # State is already a (row, col) coordinate tuple
        row, col = state

        # Actions: 0="up", 1="right", 2="down", 3="left"
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            col = min(self.grid_cols - 1, col + 1)
        elif action == 2:
            row = min(self.grid_rows - 1, row + 1)
        elif action == 3:
            col = max(0, col - 1)

        next_state = (row, col)

        # Calculate Reward and Terminal Status
        is_terminal = False
        
        # Check if next state is an obstacle
        if next_state in self.obstacles:
            reward = -1
            self.obstaclesCount += 1
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
        
        for row in range(self.grid_rows):
            row_str = ""
            for col in range(self.grid_cols):
                state = (row, col)
                
                if state == self.goal_state:
                    row_str += " 🏁 \t"
                elif state == self.start_state:
                    row_str += " 🤖 \t"
                elif state in self.obstacles:
                    row_str += " 🧱 \t"
                else:
                    # If state not visited or all values are 0
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
        for row in range(self.grid_rows):
            row_str = ""
            for col in range(self.grid_cols):
                state = (row, col)
                
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
                        
                        # If the best move is 0.0 but a wall was hit, show the negative value!
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
        for row in range(self.grid_rows):
            row_str = ""
            for col in range(self.grid_cols):
                state = (row, col)
                
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
        for row in range(self.grid_rows):
            row_str = ""
            for col in range(self.grid_cols):
                state = (row, col)
                
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
        max_steps = (self.grid_rows * self.grid_cols) * 2  # Allow more steps to navigate obstacles

        # Trace the best actions from start to finish
        while not is_terminal and steps < max_steps:
            if curr_state not in Q:
                # State not visited, can't determine best action
                break
            best_action = np.argmax(Q[curr_state])
            next_state, _, is_terminal = self.take_step(curr_state, best_action)
            path.append(next_state)
            curr_state = next_state
            steps += 1

        if curr_state != self.goal_state:
            print("<!> Warning: Agent got stuck and didn't reach the goal.")

        # Print the visual grid
        for row in range(self.grid_rows):
            row_str = ""
            for col in range(self.grid_cols):
                state = (row, col)
                
                if state == self.start_state:
                    row_str += " 🤖 \t"
                elif state == self.goal_state:
                    row_str += " 🏁 \t"
                elif state in self.obstacles:
                    row_str += " 🧱 \t"
                elif state in path:
                    row_str += " 🟢 \t" # Highlight the path with a green circle
                else:
                    row_str += " . \t"
            print(row_str)
            
        print(f"\nSteps taken: {len(path) - 1}")
        print("="*40)

    def plot_learning_curves(self, steps, rewards):
        print("Generating learning curves...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Episode via steps
        ax1.plot(steps, color='blue', alpha=0.7, linewidth=1)
        ax1.set_title("Convergence: Episode via Steps")
        ax1.set_xlabel("Episodes")
        ax1.set_ylabel("Steps to Reach Goal / Terminate")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Plot 2: Episode via reward
        ax2.plot(rewards, color='green', alpha=0.7, linewidth=1)
        ax2.set_title("Convergence: Episode via Reward")
        ax2.set_xlabel("Episodes")
        ax2.set_ylabel("Total Episode Reward")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.show()

    def simulate_qlbpw(self, start_time):

        # Initialization - Q is now a dictionary mapping state -> action values
        Q = {}

        # trackers
        optimal_path_length = 12        
        optimal_time_recorded = False   
        expected_time = 27
        track_time = True
        e_tracker = 100

        steps_per_episode = []
        rewards_per_episode = []

        # self.generate_dynamic_obstacles()

        for e in range(self.episodes):
            self.generate_dynamic_obstacles()
            # self.print_grid()
            # print("Episode 1")
            # Initialization status S
            curr_state = self.start_state

            # Other initializations
            self.gamma = 0.1 + (0.9 - 0.1) * (e / max(1, self.episodes - 1)) # gamma scales
            self.epsilon = 0.9 - (0.9 - 0.1) * (e / max(1, self.episodes - 1)) # epsilon scales DOWN
            is_terminal = False
            steps_taken = 0 # tracker
            episode_reward = 0

            while not is_terminal:
                # Behavior Policy (ε-greedy)
                action = self.epsilon_greedy(Q, curr_state)
                # self.print_agent_loc(curr_state)

                # Observe reward r and the next status s'
                next_state, reward, is_terminal = self.take_step(curr_state, action)

                # trackers
                steps_taken += 1 
                episode_reward += reward
                if ((time.time() - start_time) >= float(expected_time)) and optimal_time_recorded == False and track_time:
                    # print(f"<!> {expected_time} seconds has passed.")
                    track_time = False

                # Random Sampling
                # Initialize current state in Q if not present
                if curr_state not in Q:
                    Q[curr_state] = np.zeros(self.no_of_actions)
                
                current_q = Q[curr_state][action]
                
                if is_terminal:
                    td_target = reward
                else:
                    # Initialize next state in Q if not present
                    if next_state not in Q:
                        Q[next_state] = np.zeros(self.no_of_actions)
                    max_q_next = np.max(Q[next_state])
                    td_target = reward + self.gamma * max_q_next

                td_error = td_target - current_q

                self.er_add_experience(curr_state, action, reward, next_state, td_error)

                # Buffer Checker
                if len(self.buffer) > 0:
                    (sampled_state, sampled_action, sampled_reward, 
                     sampled_next_state, sampled_td_error, 
                     sampled_idx, adjusted_lr) = self.adjust_learning_rate()
                    # Prioritized weight update Q
                    Q = self.er_update(Q, sampled_state, sampled_action, sampled_reward, 
                                       sampled_next_state, sampled_td_error, 
                                       sampled_idx, adjusted_lr)

                # s <- s'
                curr_state = next_state
                # self.print_q_table(Q)

            # == Graph ==
            steps_per_episode.append(steps_taken)
            rewards_per_episode.append(episode_reward)

            # tracker
            if curr_state == self.goal_state and steps_taken == optimal_path_length:
                if not optimal_time_recorded:
                    time_to_optimal = time.time() - start_time
                    print(f"<!> Optimal path of {optimal_path_length} steps achieved at episode {e}! Time taken: {time_to_optimal:.2f} seconds")
                    optimal_time_recorded = True
            
            # Episode tracker - prints progress every 100 episodes
            if (e + 1) % e_tracker == 0:
                elapsed = time.time() - start_time
                self.print_q_table(Q)
                # self.print_actions(Q)
                # self.print_optimal_path(Q)
                print(f"Goal Found: {self.goalCount}")
                print(f"Obstacles Encountered: {self.obstaclesCount}")
                print(f"Episode {e + 1}/{self.episodes} | Elapsed: {elapsed:.2f}s | Steps: {steps_taken}")
                self.goalCount = 0
                self.obstaclesCount = 0

        # tracker
        self.print_actions(Q)
        self.print_q_table(Q)
        self.print_optimal_path(Q)
        print(f"Total Episodes: {self.episodes}")
        # self.plot_learning_curves(steps_per_episode, rewards_per_episode)

if __name__ == "__main__":

    # Freeze the randomness for consistent testing
    # 42 -> 386
    # random.seed(42)
    # np.random.seed(42)

    # TODO: Modify environment (0, 0)
    # TODO: Create several grid environment

    a = QLBPW(
        episodes=1000, 
        alpha=0.1, 
        gamma=0.9, 
        epsilon=0.9, 
        beta=0.3,
        dynamic_obs=True,
        num_dynamic_obs=3
    )
    print("Starting simulation wib...")
    start_time = time.time() # Start the stopwatch
    
    a.simulate_qlbpw(start_time) # <-- Pass the stopwatch in
    
    end_time = time.time() # Stop the stopwatch
    
    elapsed_time = end_time - start_time
    # print(f"Simulation finished in {elapsed_time:.2f} seconds!")