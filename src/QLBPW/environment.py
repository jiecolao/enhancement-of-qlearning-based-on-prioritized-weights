import time
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
        
        self.rows = grid
        self.columns = grid
        self.start = start
        self.goal = goal

        self.enable_obs = enable_obs
        self.obstacles = obstacles
        self.num_dynamic_obs = num_dynamic_obs
        
    def simulate(self):
        Q = {}

        for e in range(self.episodes):
            self._generate_obstacles()

            curr_state = self.start

            self.agent.gamma = 0.1 + (0.9 - 0.1) * (e / max(1, self.episodes - 1)) # gamma scales
            self.agent.epsilon = 0.9 - (0.9 - 0.1) * (e / max(1, self.episodes - 1)) # epsilon scales DOWN
            
            is_terminal = False
            
            while not is_terminal:
                action = self.agent._epsilon_greedy(Q, curr_state)

                next_state, reward, is_terminal = self._move(curr_state, action)

                if curr_state not in Q:
                    Q[curr_state] = np.zeros(self.agent.no_of_actions)

                current_q = Q[curr_state][action]

                if is_terminal:
                    td_target = reward
                else:
                    if next_state not in Q:
                        Q[next_state] = np.zeros(self.agent.no_of_actions)

                    max_q_next = np.max(Q[next_state])
                    td_target = reward + self.agent.gamma * max_q_next

                td_error = td_target - current_q

                self.agent.experience._add_experience(curr_state, action, reward, next_state, td_error)

                if len(self.agent.experience.buffer) > 0:
                    (sampled_state, sampled_action, sampled_reward, 
                     sampled_next_state, sampled_td_error, 
                     sampled_idx, adjusted_lr) = self.agent.experience._sample()
                    # Prioritized weight update Q
                    Q = self.agent._update_q_table(Q, sampled_state, sampled_action, sampled_reward, 
                                       sampled_next_state, self.goal, sampled_td_error, 
                                       sampled_idx, adjusted_lr, self.obstacles)
                    
                curr_state = next_state
                
                # print(f"Episode: {e}")
                # print(f"Agent: {curr_state}")

    def _generate_obstacles(self):
        if not self.enable_obs:
            return
        
        # self.obstacles.clear()

        obs = 0
        while obs < self.num_dynamic_obs:

            x = random.randint(0, self.columns - 1)
            y = random.randint(0, self.rows - 1)
            rand_state = (x, y)
            
            if (rand_state != self.start_state and 
                rand_state != self.goal_state and 
                rand_state not in self.obstacles):
                
                self.obstacles.append(rand_state)
                obs += 1

    def _move(self, state, action: int):
        x, y = state

        if action == 0:
            y = max(0, y - 1)
        elif action == 1:
            x = min(self.columns - 1, x + 1)
        elif action == 2:
            y = min(self.rows - 1, y + 1)
        elif action == 3:
            x = max(0, x - 1)

        next_state = (x, y)

        is_terminal = False
        
        if next_state in self.obstacles:
            reward = -1
            next_state = state
        elif next_state == self.goal:
            reward = 1
            is_terminal = True
        else:
            reward = 0

        return next_state, reward, is_terminal

    def _generate_base_obstacles(grid_size, 
                                 num_obstacles, 
                                 start_state, 
                                 goal_state, 
                                 seed=None, 
                                 blocked=None):
        rng = random.Random(seed)
        blocked_states = set(blocked or [])
        blocked_states.update([start_state, goal_state])

        all_states = [(x, y) for x in range(grid_size) for y in range(grid_size)]
        candidates = [state for state in all_states if state not in blocked_states]

        if num_obstacles > len(candidates):
            raise ValueError("num_obstacles exceeds available free cells")

        return set(rng.sample(candidates, num_obstacles))

if __name__ == "__main__":
    Environment().simulate()