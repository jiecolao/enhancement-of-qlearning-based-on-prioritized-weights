from __future__ import annotations
from typing import TYPE_CHECKING
from datetime import datetime
from .agent import Agent
import numpy as np
import tracemalloc
import os
import glob
import time

if TYPE_CHECKING: from environment import Environment

class EnvironmentTracker:

    def __init__(self, agent: Agent, env: Environment):
        self.agent = agent
        self.env = env

        self.pos = 0
        self.steps = 0
        self.rewards = 0
        self.pos_rewards = 0
        self.neg_rewards = 0
        self.obstacle_encountered = 0
        self.goal_count = 0
        self.steps_per_ep = 0
        self.rewards_per_ep = 0

        self.path_per_ep = []
        
        self.log_folder = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "QLBPW",
            "training_logs",
        )
        os.makedirs(self.log_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"qlbpw_{timestamp}.txt"
        self.full_log_path = os.path.join(self.log_folder, self.log_filename)
        self._manage_log_limit(max_logs=10)
        self._print_and_log(self._agent_details())

    def _agent_details(self):
        return (
            "\n" + "="*40 + "\n"
            "AGENT INITIALIZED\n"
            + "="*40 + "\n"
            f"{'Agent type:':<35}| {type(self.agent).__name__}\n"
            f"{'Initial learning rate (alpha):':<35}| {self.agent.init_alpha}\n"
            f"{'Learning rate (alpha):':<35}| {self.agent.alpha}\n"
            f"{'Discount factor (gamma):':<35}| {self.agent.gamma}\n"
            f"{'Beta:':<35}| {self.agent.beta}\n"
            f"{'Epsilon:':<35}| {self.agent.e}\n"
            f"{'Minimum epsilon:':<35}| {self.agent.e_min}\n"
            f"{'Epsilon decay:':<35}| {self.agent.e_decay}\n"
            f"{'Actions:':<35}| {self.agent.no_of_actions}\n"
            f"{'Replay buffer capacity:':<35}| {self.agent.max_buffer}\n"
            f"{'Replay batch size:':<35}| {self.agent.batch_size}\n"
            f"{'Q-table states at initialization:':<35}| {len(self.agent.Q)}\n"
            + "="*40
        )

    def _manage_log_limit(self, max_logs):
        search_pattern = os.path.join(self.log_folder, "qlbpw_*.txt")
        existing_logs = glob.glob(search_pattern)
        
        existing_logs.sort(key=os.path.getmtime)
        
        while len(existing_logs) >= max_logs:
            oldest_log = existing_logs.pop(0) 
            os.remove(oldest_log)             

    def _print_and_log(self, text):
        print(text)
        with open(self.full_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(text + "\n")

    def print_live_grid(self, agent_pos):
            grid_lines = ["", "="*40, "ENVIRONMENT", "="*40]
            for y in range(self.env.grid_rows):
                row_str = ""
                for x in range(self.env.grid_cols):
                    state = (x, y)
                    if state == agent_pos:
                        row_str += " A "
                    elif state == self.env.start_state:
                        row_str += " S "
                    elif state == self.env.end_state:
                        row_str += " G "
                    elif state in self.env.obstacles:
                        row_str += " # "
                    else:
                        row_str += " . "
                grid_lines.append(row_str)
            grid_lines.append("="*40)
            self._print_and_log("\n".join(grid_lines))

    def print_optimal_path(self):
        curr_state = self.env.start_state
        path = [curr_state]
        is_terminal = False
        steps = 0
        max_steps = (self.env.grid_rows * self.env.grid_cols) * 2

        while not is_terminal and steps < max_steps:
            if curr_state not in self.agent.Q:
                break
            best_action = np.argmax(self.agent.Q[curr_state])
            next_state, _, is_terminal = self.env.take_step(curr_state, best_action)
            path.append(next_state)
            curr_state = next_state
            steps += 1

        path_lines = ["", "="*40, "OPTIMAL PATH", "="*40]
        if curr_state != self.env.end_state:
            path_lines.append("<!> Warning: Agent got stuck and didn't reach the goal.")

        for y in range(self.env.grid_rows):
            row_str = ""
            for x in range(self.env.grid_cols):
                state = (x, y)
                
                if state == self.env.start_state:
                    row_str += " A "
                elif state == self.env.end_state:
                    row_str += " G "
                elif state in self.env.obstacles:
                    row_str += " # "
                elif state in path:
                    row_str += " + "
                else:
                    row_str += " . "
            path_lines.append(row_str)

        path_lines.extend(("", f"Steps taken: {len(path) - 1}", "="*40))
        self._print_and_log("\n".join(path_lines))

    def print_episode_summary(
            self, 
            curr_ep, 
            max_ep, 
            ep_tracker,
            elapsed,
            max_steps,
            epsilon
        ):
        current, peak = tracemalloc.get_traced_memory()

        summary_text = (
            f"===== EPISODE {curr_ep}/{max_ep} SUMMARY =====\n"
            f"{'Epsilon:':<30}| {epsilon:.3f}\n"
            f"{f'Steps per {ep_tracker} episode:':<30}| {self.steps_per_ep} / {max_steps*max_ep}\n"
            f"{f'Rewards per {ep_tracker} episode:':<30}| {self.rewards_per_ep}\n"
            f"{f'{ep_tracker} Episode Completion Time:':<30}| {elapsed:.2f} seconds\n"
            f"{'Memory usage:':<30}| Current: {current / (1024 * 1024):.2f} MB, Peak: {peak / (1024 * 1024):.2f} MB\n"
            f"{'Total Steps:':<30}| {self.steps}\n"
            f"{'Total Obstacles Encountered:':<30}| {self.obstacle_encountered}\n"
            f"{'Total Goals:':<30}| {self.goal_count}\n"
            f"{'Total Rewards:':<30}| {self.rewards}\n"
            f"{' ├── Positive Rewards:':<30}| {self.pos_rewards}\n"
            f"{' └── Negative Rewards:':<30}| {self.neg_rewards}\n"
        )

        self._print_and_log(summary_text)

        self.steps_per_ep = 0
        self.rewards_per_ep = 0

    def print_total_summary(self, start_time):
        elapsed_time = time.time() - start_time

        text = (
            f"\nQLBPW Total Runtime: {elapsed_time:.2f} seconds\n"
            f"QLBPW Total Rewards: {self.env.tracker.rewards}"
        )

        self._print_and_log(text=text)

if __name__ == "__main__":
    tracemalloc.start()
    a = EnvironmentTracker(0, 0, 0, 0, 0, 0, 0)
    a.print_episode_summary(1, 2, 1)