from collections import deque
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

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
        self.max_priority = 1.0

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        collision=0.0,
        goal=0.0,
        distance_progress=0.0
    ):
        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "collision": collision,
            "goal": goal,
            "distance_progress": distance_progress,
            "priority": self.max_priority
        }

        self.buffer.append(transition)

    def sample(self, priority_alpha):
        priorities = np.array(
            [transition["priority"] for transition in self.buffer],
            dtype=np.float64
        )

        scaled_priorities = priorities ** priority_alpha
        probabilities = scaled_priorities / scaled_priorities.sum()

        indices = np.random.choice(
            len(self.buffer),
            size=self.batch_size,
            replace=False,
            p=probabilities
        )

        batch = [self.buffer[i] for i in indices]

        return batch, indices, probabilities[indices]

    def importance_weights(self, probabilities, beta):
        n = len(self.buffer)

        weights = (n * probabilities) ** (-beta)
        weights /= weights.max()

        return torch.FloatTensor(weights).unsqueeze(1)

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            priority = float(priority)

            self.buffer[index]["priority"] = priority

            self.max_priority = max(
                self.max_priority,
                priority
            )

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate,
        gamma,
        priority_alpha,
        beta_start,
        beta_end,
        e,
        e_min,
        e_decay,
        max_buffer,
        batch_size,
        target_sync_freq,
        collision_weight=1.0,
        goal_weight=2.0,
        distance_weight=0.5,
    ):
        self.state_dim = state_dim
        self.no_of_actions = action_dim

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.priority_alpha = priority_alpha

        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.e = e
        self.e_min = e_min              # Epsilon Minimum
        self.e_decay = e_decay          # Epsilon Decaying Rate

        self.collision_weight = collision_weight
        self.goal_weight = goal_weight
        self.distance_weight = distance_weight

        self.main_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)

        self.target_net.load_state_dict(self.main_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.main_net.parameters(),
            lr=self.learning_rate
        )

        self.criterion = nn.SmoothL1Loss(reduction="none")              # Huber Loss

        self.memory = ReplayBuffer(max_buffer=max_buffer, batch_size=batch_size)
        self.batch_size = batch_size        # The Number of Experiences To Be Sampled
        self.max_buffer = max_buffer        # Max Number of Stored Experiences
        self.target_sync_freq = target_sync_freq  # How often the target network should sync with main 


    def save(self, agent_name, save_memory=True):
        filepath = "src/EQLBPW/trained_agents"
        os.makedirs(filepath, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if agent_name:
            filename = f"{agent_name}_{timestamp}.pth"
        else: 
            filename = f"log_{timestamp}.pth"

        full_path = os.path.join(filepath, filename)

        checkpoint = {
            'main_net_state_dict': self.main_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.e,
            'memory': list(self.memory.buffer) if save_memory else None
        }
        torch.save(checkpoint, full_path)
        print(f"Agent saved successfully to '{filepath}' (Memory saved: {save_memory})")

    def load(self, filepath="dqn_agent.pth", load_memory=True):
        checkpoint = torch.load(filepath)
        
        self.main_net.load_state_dict(checkpoint['main_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.e = checkpoint.get('epsilon', self.e_min)
        
        if load_memory and checkpoint.get('memory') is not None:
            self.memory.buffer.clear()
            for transition in checkpoint['memory']:
                self.memory.buffer.append(transition)
                
        self.main_net.train()
        self.target_net.eval()
        print(f"Agent loaded successfully from '{filepath}'")

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

    def update_beta(self, progress):
        progress = min(max(progress, 0.0), 1.0)
        self.beta = (self.beta_start + progress * (self.beta_end - self.beta_start))

    
    def calculate_priority(
        self,
        td_error,
        collision,
        goal,
        distance_progress
    ):
        weight = (
            1.0
            + self.collision_weight * collision
            + self.goal_weight * goal
            + self.distance_weight *
            torch.clamp(distance_progress, min=0.0)
        )

        priority = (
            td_error.abs() + 1e-5
        ) * weight

        return priority
    
    def update(self):
        if len(self.memory) < self.batch_size:
            return

        batch, indices, probabilities = self.memory.sample(
            self.priority_alpha
        )

        states = torch.FloatTensor(
            np.array([t["state"] for t in batch])
        )

        actions = torch.LongTensor(
            [t["action"] for t in batch]
        ).unsqueeze(1)

        rewards = torch.FloatTensor(
            [t["reward"] for t in batch]
        ).unsqueeze(1)

        next_states = torch.FloatTensor(
            np.array([t["next_state"] for t in batch])
        )

        dones = torch.FloatTensor(
            [t["done"] for t in batch]
        ).unsqueeze(1)

        collisions = torch.FloatTensor(
            [t["collision"] for t in batch]
        ).unsqueeze(1)

        goals = torch.FloatTensor(
            [t["goal"] for t in batch]
        ).unsqueeze(1)

        distance_progress = torch.FloatTensor(
            [t["distance_progress"] for t in batch]
        ).unsqueeze(1)

        weights = self.memory.importance_weights(
            probabilities,
            self.beta
        )

        q_current = self.main_net(states).gather(
            1,
            actions
        )

        with torch.no_grad():
            best_next_actions = (self.main_net(next_states).argmax(1, keepdim=True))

            q_next = (self.target_net(next_states).gather(1, best_next_actions))

            q_target = (rewards + (1 - dones) * self.gamma * q_next)

        td_error = q_target - q_current

        element_loss = self.criterion(q_current, q_target)

        loss = (weights * element_loss).mean()

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.main_net.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        with torch.no_grad():
            priorities = self.calculate_priority(
                td_error,
                collisions,
                goals,
                distance_progress
            )

        self.memory.update_priorities(
            indices,
            priorities.squeeze(1).cpu().numpy()
        )

    def _test(self):
        print("agent.py acccessed!")