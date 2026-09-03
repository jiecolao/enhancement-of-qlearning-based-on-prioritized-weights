import tracemalloc
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

class Visualizer:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def dqn_visualize_learned_path(self, title="DQN Optimal Path"):
        """Visualize the optimal path learned by a DQN using Matplotlib"""
        curr_state = self.env.start_state
        path = [curr_state]
        is_terminal = False
        steps = 0
        max_steps = (self.env.grid_rows * self.env.grid_cols) * 2

        self.agent.main_net.eval() 

        while not is_terminal and steps < max_steps:
            state_t = torch.FloatTensor(curr_state).unsqueeze(0) 
            
            with torch.no_grad(): 
                best_action = self.agent.main_net(state_t).argmax().item() 
                
            next_state, _, is_terminal = self.env.take_step(curr_state, best_action)
            
            if np.array_equal(curr_state, next_state):
                print("Agent got stuck! Ending path tracing.")
                break
                
            path.append(next_state)
            curr_state = next_state
            steps += 1

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 10))

        # Create grid background
        ax.set_xlim(-0.5, self.env.grid_cols - 0.5)
        ax.set_ylim(self.env.grid_rows - 0.5, -0.5)  # Inverted Y-axis for proper orientation
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Draw obstacles
        for obs in self.env.obstacles:
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
        ax.scatter(*self.env.end_state, c='red', s=300, marker='*', label='Goal', zorder=5)

        # Draw start position
        ax.scatter(*self.env.start_state, c='blue', s=200, marker='o', label='Start', zorder=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()

        print(f"Path length: {len(path) - 1} steps")
        plt.show()