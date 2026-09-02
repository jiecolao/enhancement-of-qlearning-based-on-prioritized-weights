from collections import deque
import random
import numpy as np


class ReplayBuffer:
    def __init__(self, max_buffer, batch_size):
        self.buffer = deque(maxlen=max_buffer)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, td_error):
        self.buffer.append([state, action, reward, next_state, td_error])

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, td_error = zip(*batch)
        return (
            states,
            actions,
            rewards, 
            next_states,
            td_error
        )

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(
            self, 
            alpha, 
            gamma, 
            beta,
            e, 
            e_min, 
            e_decay,        
            no_of_states, 
            no_of_actions,
            max_buffer,
            batch_size, 
    ):
        self.init_alpha = alpha
        self.alpha = alpha              # Learning Rate   
        self.gamma = gamma              # Discount Factor
        self.beta = beta                # ???

        self.e = e                      # Epsilon
        self.e_min = e_min              # Epsilon Minimum
        self.e_decay = e_decay          # Epsilon Decaying Rate

        self.no_of_actions = no_of_actions
        self.Q = {}

        self.memory = ReplayBuffer(max_buffer=max_buffer, batch_size=batch_size)
        self.batch_size = batch_size    # The Number of Experiences To Be Sampled
        self.max_buffer = max_buffer    # Max Number of Stored Experiences

    def update_Q(
            self, 
            state, 
            action, 
            reward, 
            next_state, 
            td_error, 
            sampled_idx, 
            adjusted_lr,
            end_state,
            obstacles
    ):
        if not self.memory.buffer:
            return self.Q

        if state not in self.Q:
            self.Q[state] = np.zeros(self.no_of_actions)
        
        current_q = self.Q[state][action]
        
        if next_state == end_state or next_state in obstacles:
            td_target = reward
        else:
            if next_state not in self.Q:
                self.Q[next_state] = np.zeros(self.no_of_actions)
            max_q_next = np.max(self.Q[next_state])
            td_target = reward + self.gamma * max_q_next
        
        new_td_error = td_target - current_q 
        
        self.Q[state][action] = (1 - adjusted_lr) * current_q + (adjusted_lr * td_target)
        
        self.memory.buffer[sampled_idx][4] = float(new_td_error)
        
        return self.Q

    def epsilon_greedy(self, state):
        action = random.random()

        if action < self.e:
            return random.randrange(self.no_of_actions)
        else: 
            Q_values = self.Q.get(state, np.zeros(self.no_of_actions))
            return np.argmax(Q_values)

    def adjust_lr(self):
        b = len(self.memory)

        errors = np.array([abs(exp[4]) for exp in self.memory.buffer])
        
        sorted_indices = np.argsort(-errors)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, b + 1)

        p_j_unnormalized = 1.0 / ranks
        p_j = p_j_unnormalized / np.sum(p_j_unnormalized)
        
        sampled_idx = np.random.choice(b, p=p_j)
        state, action, reward, next_state, td_error = self.memory.buffer[sampled_idx]
        
        p_sampled = p_j[sampled_idx]
        adjusted_lr = self.init_alpha / ((b * p_sampled) ** self.beta) 

        return state, action, reward, next_state, td_error, sampled_idx, adjusted_lr
