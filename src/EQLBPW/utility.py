import tracemalloc
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def print_memory_stats(label):
    current, peak = tracemalloc.get_traced_memory()
    print(f"{label} | Current: {current / (1024 * 1024):.2f} MB | Peak: {peak / (1024 * 1024):.2f} MB\n")

def visualize_learned_path_dqn(self, agent, title="DQN Optimal Path"):
    """Visualize the optimal path learned by a DQN using Matplotlib"""
    # Trace the optimal path using the DQN Policy Network
    curr_state = self.start_state
    path = [curr_state]
    is_terminal = False
    steps = 0
    max_steps = (self.grid_rows * self.grid_cols) * 2

    # Set network to evaluation mode (optional but good practice)
    agent.main_net.eval() 

    while not is_terminal and steps < max_steps:
        # 1. Format the state for the neural network
        state_t = torch.FloatTensor(curr_state).unsqueeze(0) 
        
        # 2. Get the best action from the network without tracking gradients
        with torch.no_grad(): 
            best_action = agent.main_net(state_t).argmax().item() 
            
        # 3. Take the step in the environment
        next_state, _, is_terminal = self._take_step(curr_state, best_action)
        
        # 4. Break if the agent is stuck in a loop (state hasn't changed)
        if np.array_equal(curr_state, next_state):
            print("Agent got stuck! Ending path tracing.")
            break
            
        path.append(next_state)
        curr_state = next_state
        steps += 1

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create grid background
    ax.set_xlim(-0.5, self.grid_cols - 0.5)
    ax.set_ylim(self.grid_rows - 0.5, -0.5)  # Inverted Y-axis for proper orientation
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Draw obstacles
    for obs in self.obstacles:
        rect = patches.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1,
                                linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(rect)

    # Draw path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'g-', linewidth=2, alpha=0.6, label='Learned Path')
        ax.scatter(path_x, path_y, c='green', s=20, alpha=0.5)

    # Draw goal
    ax.scatter(*self.end_state, c='red', s=300, marker='*', label='Goal', zorder=5)

    # Draw start position
    ax.scatter(*self.start_state, c='blue', s=200, marker='o', label='Start', zorder=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    print(f"Path length: {len(path) - 1} steps")
    plt.show()
