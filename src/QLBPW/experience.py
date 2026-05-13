import numpy as np

class Experience:

    def __init__(self, 
            max_buffer,
            initial_alpha,
            beta
            ):
        self.max_buffer = max_buffer
        self.initial_alpha = initial_alpha
        self.beta = beta

        self.buffer = []
        self.pos = 0

    def _add_experience(self, 
                        state, 
                        action, 
                        reward, 
                        next_state, 
                        td_error):
        exp = [state, int(action), float(reward), next_state, float(td_error)]

        if len(self.buffer) < self.max_buffer:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp
        
        self.pos = (self.pos + 1) % self.max_buffer
    
    def _sample(self):
        if not self.buffer:
            return None
        
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