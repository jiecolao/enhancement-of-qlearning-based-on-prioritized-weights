import numpy as np

class Agent:

    """ ENHANCED Q-LEARNING BASED ON PRIORITIZED WEIGHTS """

    def __init__(
            self,
            episodes,       # Number of iterations
            alpha,          # Learning Rate   
            gamma,          # Discount Factor
            beta,           # 

            e,              # Epsilon
            e_min,          # Epsilon Minimum
            e_decay,        # Epsilon Decaying Rate
            
            
            batch_size,     # 
            max_buffer,     # 
            ):
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

        self.e = e
        self.e_min = e_min
        self.e_decay = e_decay

        self.batch_size = batch_size
        self.max_buffer = max_buffer      

    def _adjust_alpha(self):
        pass

    def _e_greedy(self):
        pass

    """"""
