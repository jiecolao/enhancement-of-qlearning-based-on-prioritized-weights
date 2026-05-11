import random
import numpy as np
from QLBPW.agent import Agent

class Environment:

    def __init__(self,
            agent: Agent,

            grid: int = 9,
            start: tuple = (0, 0),
            goal: tuple = (0, 1),

            enable_obs: bool = False,
            obstacles: list = [], 
            num_dynamic_obs: int = 5
            ):
        self.agent = agent
        self.actions = agent.actions
        self.episodes = agent.episodes
        
        self.grid = grid
        self.rows = grid-1
        self.columns = grid-1
        self.start = start             # Should cater both list/literal
        self.goal = goal              # Should cater both list/literal

        self.enable_obs = enable_obs
        self.obstacles = obstacles
        self.num_dynamic_obs = num_dynamic_obs
        
    def simulate(self):
        Q = {}

        for e in range(self.episodes):
            self._generate_obstacles()

            curr_state = self.start

            self.gamma = 0.1 + (0.9 - 0.1) * (e / max(1, self.episodes - 1)) # gamma scales
            self.epsilon = 0.9 - (0.9 - 0.1) * (e / max(1, self.episodes - 1)) # epsilon scales DOWN
            
            is_terminal = False
            
            while not is_terminal:
                action = self.agent._epsilon_greedy(Q, curr_state)

                next_state, reward, is_terminal = self._move(curr_state, action)

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

                self.agent._experience_replay(curr_state, action, reward, next_state, td_error)

                if len(self.buffer) > 0:
                    (sampled_state, sampled_action, sampled_reward, 
                     sampled_next_state, sampled_td_error, 
                     sampled_idx, adjusted_lr) = self.agent._adjust_gamma()
                    # Prioritized weight update Q
                    Q = self.agent._update_q_table(Q, sampled_state, sampled_action, sampled_reward, 
                                       sampled_next_state, sampled_td_error, 
                                       sampled_idx, adjusted_lr, self.obstacles)
                    
                curr_state = next_state

    def _generate_obstacles(self):
        if not self.enable_obs:
            return
        
        # self.obstacles.clear()

        obs = 0
        while obs < self.num_dynamic_obs:

            x = random.randint(0, self.columns)
            y = random.randint(0, self.rows)
            rand_state = (x, y)
            
            if (rand_state != self.start_state and 
                rand_state != self.goal_state and 
                rand_state not in self.obstacles):
                
                self.obstacles.append(rand_state)
                obs += 1

    def _move(self, state, action: int):
        x, y = state
        
        if action not in self.actions:
            return

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

if __name__ == "__main__":
    Environment().simulate()