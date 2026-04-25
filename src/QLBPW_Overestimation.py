import numpy as np
import matplotlib.pyplot as plt

class PrioritizedQAgent:
    def __init__(self, state_size, action_size, b=500, beta=0.3, alpha=0.1, gamma=0.9, epsilon=0.9):
        self.state_size = state_size
        self.action_size = action_size
        self.b = b
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))
        self.memory = []

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state, :])

    def store_transition(self, state, action, reward, next_state):
        q_now = self.q_table[state, action]
        q_next_max = np.max(self.q_table[next_state, :])
        td_target = reward + self.gamma * q_next_max
        td_error = td_target - q_now
        
        self.memory.append({'transition': (state, action, reward, next_state), 
                            'error': abs(td_error)})
        
        if len(self.memory) > self.b:
            self.memory.pop(0)

    def learn(self):
        if len(self.memory) < 2:
            return
            
        self.memory.sort(key=lambda x: x['error'], reverse=True)
        ranks = np.arange(1, len(self.memory) + 1)
        probabilities = 1.0 / ranks
        probabilities /= np.sum(probabilities)
        
        sampled_index = np.random.choice(len(self.memory), p=probabilities)
        sample = self.memory[sampled_index]
        state, action, reward, next_state = sample['transition']
        p_j = probabilities[sampled_index]
        
        alpha_j = self.alpha / ((self.b * p_j) ** self.beta)
        
        q_now = self.q_table[state, action]
        q_next_max = np.max(self.q_table[next_state, :])
        td_target = reward + self.gamma * q_next_max
        
        self.q_table[state, action] = (1 - alpha_j) * q_now + alpha_j * td_target
        self.memory[sampled_index]['error'] = abs(td_target - self.q_table[state, action])

def run_overestimation_demo():
    np.random.seed(42)
    state_size = 1
    action_size = 10
    # Use epsilon=1.0 to ensure continuous uniform exploration 
    agent = PrioritizedQAgent(state_size=state_size, action_size=action_size, gamma=0.9, alpha=0.1, epsilon=1.0)
    
    max_q_values = []
    
    episodes = 500
    steps_per_episode = 10
    
    for e in range(episodes):
        state = 0
        for _ in range(steps_per_episode):
            action = agent.choose_action(state)
            # The environment is designed such that the true Q-value for all actions is 0
            # We provide noisy rewards with a mean of 0 and std deviation of 1
            reward = np.random.normal(0, 1)
            next_state = 0
            
            agent.store_transition(state, action, reward, next_state)
            agent.learn()
            
        max_q_values.append(np.max(agent.q_table[0]))
        
    plt.figure(figsize=(10, 6))
    plt.plot(max_q_values, label="Max Q-value estimate", color='red')
    plt.axhline(y=0, color='blue', linestyle='--', label="True Max Q-value", linewidth=2)
    plt.title("Overestimation Bias in QLBPW over 500 Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Maximum Q-value Estimate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_overestimation_demo()