import numpy as np
import random

class QLBPW():
    def __init__(self, episodes, alpha, gamma, epsilon, beta):
        self.episodes = episodes
        self.initial_alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta = beta # superparam 

        # Environment
        self.grid_rows = 9
        self.grid_cols = 9
        actions = ["up", "right", "down", "left"]
        self.no_of_actions = len(actions)
        self.no_of_states = self.grid_rows * self.grid_cols
        self.start_state = 0
        self.goal_state = 80
        self.obstacles = [
        ]

        # Prioritized Experience Replay (Empirical experience)
        self.buffer = []
        self.td_errors = []
        self.maxcap = 5000

    def epsilon_greedy(self, Q, state):
        a = random.random()
        if a < self.epsilon: # exploration
            return random.randrange(self.no_of_actions)
        else: # exploitation
            return np.argmax(Q[state])
    
    def get_reward(self):
        pass

    def adjust_learning_rate(self):
        # print(self.buffer)
        b = len(self.buffer)

        ranks = np.arange(1, b + 1)
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
        if len(self.buffer) >= self.maxcap:
            self.buffer.pop(-1)

        self.buffer.append([state, action, reward, next_state, td_error])

        self.buffer.sort(key=lambda x: abs(x[4]), reverse=True)

    def er_update(self, Q, state, action, reward, next_state, td_error, sampled_idx, adjusted_lr):
        if not self.buffer:
            return Q

        # 3. Calculate TD Target and new TD Error (Equations 6, 7, and 8)
        current_q = Q[state, action]
        max_q_next = np.max(Q[next_state])
        
        # Calculate the TD target
        td_target = reward + self.gamma * max_q_next
        
        # Calculate the new TD error (delta_j)
        new_td_error = td_target - current_q 
        
        # 4. Update the Q-Table (Equation 9)
        # We replace the standard alpha with our adjusted_lr (a_j)
        Q[state, action] = (1 - adjusted_lr) * current_q + (adjusted_lr * td_target)
        
        # 5. Update the error in the buffer and re-sort
        self.buffer[sampled_idx][4] = new_td_error
        self.buffer.sort(key=lambda x: abs(x[4]), reverse=True)
        
        return Q
    
    def take_step(self, state, action):
        # Convert 1D state index to 2D grid coordinates (row, col)
        row = state // self.grid_cols
        col = state % self.grid_cols

        # Actions: 0="up", 1="right", 2="down", 3="left"
        if action == 0:
            row = max(0, row - 1)
        elif action == 1:
            col = min(self.grid_cols - 1, col + 1)
        elif action == 2:
            row = min(self.grid_rows - 1, row + 1)
        elif action == 3:
            col = max(0, col - 1)

        # Convert back to 1D state index
        next_state = (row * self.grid_cols) + col

        # Calculate Reward and Terminal Status
        is_terminal = False
        
        if next_state == self.goal_state:
            reward = 1
            is_terminal = True
        elif next_state in self.obstacles: 
            reward = -1 # Fixing the paper's typo!
            is_terminal = True # Often, hitting an obstacle ends the episode
        else:
            reward = 0

        return next_state, reward, is_terminal

    def print_q_table(self, Q):
        print("Learned Policy (Best Actions):")
        action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        
        for row in range(self.grid_rows):
            row_str = ""
            for col in range(self.grid_cols):
                state = (row * self.grid_cols) + col
                
                if state == self.goal_state:
                    row_str += " G \t"
                elif state == self.start_state:
                    row_str += " S \t"
                elif state in self.obstacles:
                    row_str += " X \t"
                else:
                    # If all values are 0, it hasn't explored this state successfully yet
                    if np.max(Q[state]) == 0:
                        row_str += " . \t" 
                    else:
                        best_action = np.argmax(Q[state])
                        row_str += f" {action_symbols[best_action]} \t"
            print(row_str)
            
        print("\nMax Q-Values:")
        for row in range(self.grid_rows):
            row_str = ""
            for col in range(self.grid_cols):
                state = (row * self.grid_cols) + col
                
                if state == self.goal_state:
                    row_str += " GOAL \t"
                elif state == self.start_state:
                    row_str += " START \t"
                elif state in self.obstacles:
                    row_str += " OBST \t"
                else:
                    # Print the highest Q-value rounded to 2 decimal places
                    row_str += f"{np.max(Q[state]):.2f}\t"
            print(row_str)
        print("-" * 40)

    def simulate_qlbpw(self):

        Q = np.zeros((self.no_of_states, self.no_of_actions))
        
        is_terminal = False

        for e in range(self.episodes):
            self.gamma = 0.1 + (0.9 - 0.1) * (e / max(1, self.episodes - 1)) # gamma scales
            curr_state = self.start_state
            is_terminal = False

            while not is_terminal:
                # 1. Choose an action
                action = self.epsilon_greedy(Q, curr_state)

                # 2. Take the action and observe the environment
                next_state, reward, is_terminal = self.take_step(curr_state, action)

                # 3. Calculate initial TD Error to store in buffer
                current_q = Q[curr_state, action]
                max_q_next = np.max(Q[next_state])
                td_target = reward + self.gamma * max_q_next
                td_error = td_target - current_q

                # 4. Add to Experience Replay Buffer
                self.er_add_experience(curr_state, action, reward, next_state, td_error)

                # 5. Sample from buffer and update Q-table (if buffer has enough data)
                if len(self.buffer) > 0:
                    (sampled_state, sampled_action, sampled_reward, 
                     sampled_next_state, sampled_td_error, 
                     sampled_idx, adjusted_lr) = self.adjust_learning_rate()
                    
                    Q = self.er_update(Q, sampled_state, sampled_action, sampled_reward, 
                                       sampled_next_state, sampled_td_error, 
                                       sampled_idx, adjusted_lr)

                # 6. Move to the next state
                curr_state = next_state
            
        self.print_q_table(Q)

if __name__ == "__main__":
    a = QLBPW(episodes=100, alpha=0.1, gamma=0.9, epsilon=0.9, beta=0.3)
    a.simulate_qlbpw()