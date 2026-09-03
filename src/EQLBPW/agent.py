from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class QNetwork(nn.Module):
    # OBJ 3: Double Deep Q-Learning

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        # Defines how data moves in the NN
        return self.net(state)


class ReplayBuffer:
    def __init__(self, max_buffer, batch_size):
        self.buffer = deque(maxlen=max_buffer)
        self.batch_size = batch_size

    def push(self, state, action, reward, next_state, is_terminal):
        self.buffer.append((state, action, reward, next_state, is_terminal))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, is_terminal = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions).unsqueeze(1),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(is_terminal).unsqueeze(1)
            # torch.FloatTensor(dones).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(
            self, 
            # episodes, 
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
            target_sync_freq 
    ):
        self.alpha = alpha              # Learning Rate   
        self.gamma = gamma              # Discount Factor
        self.beta = beta                # ???

        self.e = e                      # Epsilon
        self.e_min = e_min              # Epsilon Minimum
        self.e_decay = e_decay          # Epsilon Decaying Rate

        self.no_of_actions = no_of_actions
        self.main_net = QNetwork(2, no_of_actions)       # Main Network
        self.target_net = QNetwork(2, no_of_actions)     # Target Network
        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(max_buffer=max_buffer, batch_size=batch_size)
        self.batch_size = batch_size        # The Number of Experiences To Be Sampled
        self.max_buffer = max_buffer        # Max Number of Stored Experiences
        self.target_sync_freq = target_sync_freq  # How often the target network should sync with main 

    def adjust_alpha(self):
        pass

    def e_greedy(self, state):
        if random.random() < self.e:
            return random.randrange(self.no_of_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.main_net(state_t).argmax().item()

    def decay_e(self):
        # OBJ 2: Decaying epsilon
        self.e = max(self.e_min, self.e * self.e_decay)

    def sync_target(self):
        self.target_net.load_state_dict(self.main_net.state_dict())
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        # Q(s, a; \theta)
        q_current = self.main_net(states).gather(1, actions)

        # r + \gamma * max_a' Q(s', a'; \theta^-)
        with torch.no_grad():
            q_next_max = self.target_net(next_states).max(1, keepdim=True)[0]
            q_target = rewards + (1 - dones) * self.gamma * q_next_max

        loss = self.criterion(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # self._decay_e()

    def _test(self):
        print("agent.py acccessed!")