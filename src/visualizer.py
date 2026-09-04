from datetime import datetime
from pathlib import Path
import tracemalloc
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import glob


class Visualizer:
    def __init__(self, agent, env, max_figures=10):
        self.agent = agent
        self.env = env

        self.max_figures = max_figures

    def _manage_figure_limit(self, figure_dir):
        image_patterns = ("*.png", "*.jpg", "*.jpeg")
        existing_figures = [
            filepath
            for pattern in image_patterns
            for filepath in glob.glob(os.path.join(figure_dir, pattern))
        ]
        existing_figures.sort(key=os.path.getmtime)

        while len(existing_figures) >= self.max_figures:
            os.remove(existing_figures.pop(0))

    def _save_to_eqlbpw(
            self, 
            title=None, 
            fig=None
        ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_dir = Path(__file__).resolve().parent / "EQLBPW" / "figures"
        os.makedirs(figure_dir, exist_ok=True)
        self._manage_figure_limit(figure_dir)

        if title:
            filename = f"{title}_{timestamp}.png"
        else:
            filename = f"figure_{timestamp}.png"
            
        filepath = os.path.join(figure_dir, filename)
        (fig or plt.gcf()).savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"\nVisualization saved to: {filepath}")

    def _save_to_qlbpw(
            self, 
            title=None, 
            fig=None
        ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_dir = Path(__file__).resolve().parent / "QLBPW" / "figures"
        os.makedirs(figure_dir, exist_ok=True)
        self._manage_figure_limit(figure_dir)

        if title:
            filename = f"{title}_{timestamp}.png"
        else:
            filename = f"figure_{timestamp}.png"
            
        filepath = os.path.join(figure_dir, filename)
        (fig or plt.gcf()).savefig(filepath, dpi=100, bbox_inches='tight')
        print(f"\nVisualization saved to: {filepath}") 

    def qlbpw_visualize_learned_path( 
            self,
            agent,
            env,
            title="Q-Learning Optimal Path",
            save_fig=False
        ):
        curr_state = env.start_state
        path = [curr_state]
        is_terminal = False
        steps = 0
        max_steps = (env.grid_rows * env.grid_cols) * 2

        while not is_terminal and steps < max_steps:
            if curr_state not in agent.Q:
                break
            best_action = np.argmax(agent.Q[curr_state])
            next_state, _, is_terminal = env.take_step(curr_state, best_action)
            path.append(next_state)
            curr_state = next_state
            steps += 1

        fig, ax = plt.subplots(figsize=(10, 10))

        ax.set_xlim(-0.5, env.grid_cols - 0.5)
        ax.set_ylim(env.grid_rows - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        for obs in env.obstacles:
            rect = patches.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1,
                                    linewidth=1, edgecolor='black', facecolor='black')
            ax.add_patch(rect)

        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'g-', linewidth=2, alpha=0.6, label='Learned Path')
            ax.scatter(path_x, path_y, c='green', s=20, alpha=0.5)

        ax.scatter(*env.end_state, c='red', s=300, marker='*', label='Goal', zorder=5)
        ax.scatter(*env.start_state, c='blue', s=200, marker='o', label='Start', zorder=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()

        print(f"Path length: {len(path) - 1} steps")
        
        if save_fig:
            self._save_to_qlbpw(title=title, fig=fig)
        plt.show()

    def eqlbpqw_visualize_learned_path(
            self, 
            title="DQN Optimal Path", 
            save_fig=False
        ):
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

        if save_fig:
            self._save_to_eqlbpw(title=title, fig=fig)
        plt.show()

    def plot_line(
            self, 
            x, 
            y, 
            title="Line Plot", 
            xlabel="X Axis", 
            ylabel="Y Axis", 
            label="Trend",
            algo=None,
            save_fig=False,
        ):
        # Arguments: Literals or Array values for 'x' and 'y'
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x, y, color="tab:blue", linewidth=2, marker="o", label=label)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.6)
        if label:
            ax.legend()
        fig.tight_layout()

        if save_fig and algo == "qlbpw":
            self._save_to_qlbpw(title=title, fig=fig)
        elif save_fig and algo == "eqlbpqw":
            self._save_to_eqlbpw(title=title, fig=fig)

        plt.show()

    # 2. Scatter Plot Template
    def plot_scatter(
            self, 
            x, 
            y, 
            title="Scatter Plot", 
            xlabel="X Axis", 
            ylabel="Y Axis", 
            color=None,
            algo=None,
            save_fig=False,
        ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        scatter = ax.scatter(x, y, c=color if color is not None else "tab:orange", alpha=0.7, edgecolors="none")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        if color is not None and len(np.unique(color)) > 1:
            fig.colorbar(scatter, ax=ax, label="Value")
        fig.tight_layout()

        if save_fig and algo == "qlbpw":
            self._save_to_qlbpw(title=title, fig=fig)
        elif save_fig and algo == "eqlbpqw":
            self._save_to_eqlbpw(title=title, fig=fig)

        plt.show()

    def plot_bar(
            self, 
            categories, 
            values, 
            title="Bar Chart", 
            xlabel="Category", 
            ylabel="Value", 
            horizontal=False,
            algo=None,
            save_fig=False
        ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if horizontal:
            ax.barh(categories, values, color="tab:green", edgecolor="black", linewidth=0.5)
            ax.set_xlabel(ylabel)
            ax.set_ylabel(xlabel)
        else:
            ax.bar(categories, values, color="tab:green", edgecolor="black", linewidth=0.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="x" if horizontal else "y", linestyle="--", alpha=0.6)
        fig.tight_layout()
        
        if save_fig and algo == "qlbpw":
            self._save_to_qlbpw(title=title, fig=fig)
        elif save_fig and algo == "eqlbpqw":
            self._save_to_eqlbpw(title=title, fig=fig)

        plt.show()

    # 4. Histogram Template
    def plot_histogram(
            self, 
            data, 
            bins=20, 
            title="Distribution", 
            xlabel="Value", 
            ylabel="Frequency",
            algo=None,
            save_fig=False
        ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(data, bins=bins, color="tab:purple", edgecolor="black", alpha=0.75)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        fig.tight_layout()
        
        if save_fig and algo == "qlbpw":
            self._save_to_qlbpw(title=title, fig=fig)
        elif save_fig and algo == "eqlbpqw":
            self._save_to_eqlbpw(title=title, fig=fig)

        plt.show()

    # 5. Box Plot Template
    def plot_boxplot(
            self, 
            data, 
            labels=None, 
            title="Box Plot", 
            ylabel="Values",
            algo=None,
            save_fig=False
        ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.boxplot(data, patch_artist=True, tick_labels=labels,
                boxprops=dict(facecolor="lightblue", color="blue"),
                medianprops=dict(color="red", linewidth=1.5))
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        fig.tight_layout()

        if save_fig and algo == "qlbpw":
            self._save_to_qlbpw(title=title, fig=fig)
        elif save_fig and algo == "eqlbpqw":
            self._save_to_eqlbpw(title=title, fig=fig)

        plt.show()

    # 6. Heatmap Template
    def plot_heatmap(
            self, 
            matrix, 
            row_labels=None, 
            col_labels=None, 
            title="Heatmap", 
            cmap="viridis",
            algo=None,
            save_fig=False
        ):
        fig, ax = plt.subplots(figsize=(7, 6))
        cax = ax.imshow(matrix, cmap=cmap, aspect="auto")
        fig.colorbar(cax, ax=ax)
        
        if row_labels is not None:
            ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
        if col_labels is not None:
            ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
        ax.set_title(title, fontsize=12, fontweight="bold")
        fig.tight_layout()

        if save_fig and algo == "qlbpw":
            self._save_to_qlbpw(title=title, fig=fig)
        elif save_fig and algo == "eqlbpqw":
            self._save_to_eqlbpw(title=title, fig=fig)

        plt.show()

    def compare_plot_line(
        self,
        x,
        series,
        title="Comparison",
        xlabel="X Axis",
        ylabel="Value",
        save_fig=False,
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))

        for label, values in series.items():
            ax.plot(x, values, marker="o", linewidth=2, label=label)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        fig.tight_layout()

        if save_fig:
            self._save_to_eqlbpw(title=title, fig=fig)

        plt.show()


if __name__=="__main__":
    vis = Visualizer(None, None)

    months = np.arange(1, 13)
    active_users = np.array([12, 15, 14, 18, 22, 25, 29, 31, 35, 38, 42, 48])

    vis.plot_line(
        x=months,
        y=active_users,
        title="Monthly Active Users Growth (2025)",
        xlabel="Month",
        ylabel="Users (in thousands)",
        label="MAU"
    )