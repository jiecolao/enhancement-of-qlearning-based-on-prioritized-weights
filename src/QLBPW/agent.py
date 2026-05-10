import numpy as np
import random

class Agent:

    def __init__(self,
            episodes: int,
            alpha: float,
            gamma: float, 
            beta: float, 
            epsilon: float, 
            
            actions: list =['up', 'right', 'down', 'left'],

            max_buffer: int = 5000,
            ):
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon

        self.actions = actions
        self.no_of_actions = len(actions)

        self.pos = 0
        self.buffer = []
        self.max_buffer = max_buffer

    def _epsilon_greedy(self, Q, state):
        a = random.random()

        if a < self.epsilon:
            # Exploration
            return random.randrange(self.no_of_actions)
        else:
            # Exploitation
            q_values = Q.get(state, np.zeros(self.no_of_actions))
            return np.argmax(q_values)

    def _adjust_gamma(self):
        pass

    def _experience_replay(self, state, action, reward, next_state, td_error):
        experience = (state, action, reward, next_state, td_error)

        if len(self.buffer) < self.max_buffer:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.pos = (self.pos + 1) % self.max_buffer
    
    def _update_q_table(self, Q, state, action, reward, next_state, td_error, sampled_idx, adjusted_lr, obstacles):
        if not self.buffer:
            return Q

        if state not in Q:
            Q[state] = np.zeros(self.no_of_actions)
        
        current_q = Q[state][action]
        
        if next_state == self.goal or next_state in obstacles:
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
