from __future__ import annotations
from typing import TYPE_CHECKING
from datetime import datetime
from .agent import Agent
import torch
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
        self.successes_per_ep = 0
        self.collisions_per_ep = 0
        self.path_lengths = []
        self.success_history = []
        self.reward_history = []
        self.steps_history = []
        self.optimality_history = []

        self.shortest_path = self.calculate_shortest_path()

        if self.shortest_path is not None:
            self.shortest_path_steps = len(self.shortest_path) - 1
        else:
            self.shortest_path_steps = None

        self.path_per_ep = []

        self.log_folder = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "EQLBPW",
            "training_logs",
        )
        os.makedirs(self.log_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"eqlbpw_{timestamp}.txt"
        self.full_log_path = os.path.join(self.log_folder, self.log_filename)
        self._manage_log_limit(max_logs=10)
        self._print_and_log(self._agent_details())

    def _agent_details(self):
        parameter_count = sum(
            parameter.numel() for parameter in self.agent.main_net.parameters()
        )
        return (
            "\n" + "="*40 + "\n"
            "AGENT INITIALIZED\n"
            + "="*40 + "\n"
            f"{'Agent type:':<30}| {type(self.agent).__name__}\n"
            f"{'State dimensions:':<30}| {self.agent.state_dim}\n"
            f"{'Learning rate:':<30}| {self.agent.learning_rate}\n"
            f"{'Discount factor (gamma):':<30}| {self.agent.gamma}\n"
            f"{'Priority alpha:':<30}| {self.agent.priority_alpha}\n"
            f"{'Beta start:':<30}| {self.agent.beta_start}\n"
            f"{'Beta end:':<30}| {self.agent.beta_end}\n"
            f"{'Current beta:':<30}| {self.agent.beta}\n"
            f"{'Epsilon:':<30}| {self.agent.e}\n"
            f"{'Minimum epsilon:':<30}| {self.agent.e_min}\n"
            f"{'Epsilon decay:':<30}| {self.agent.e_decay}\n"
            f"{'Actions:':<30}| {self.agent.no_of_actions}\n"
            f"{'Replay buffer capacity:':<30}| {self.agent.max_buffer}\n"
            f"{'Replay batch size:':<30}| {self.agent.batch_size}\n"
            f"{'Target sync frequency:':<30}| {self.agent.target_sync_freq}\n"
            f"{'Collision priority weight:':<30}| {self.agent.collision_weight}\n"
            f"{'Goal priority weight:':<30}| {self.agent.goal_weight}\n"
            f"{'Distance priority weight:':<30}| {self.agent.distance_weight}\n"
            f"{'Main network parameters:':<30}| {parameter_count}\n"
            f"{'Main network:':<30}|\n{self.agent.main_net}\n"
            + "="*40
        )

    def _manage_log_limit(self, max_logs):
        search_pattern = os.path.join(self.log_folder, "eqlbpw_*.txt")
        existing_logs = glob.glob(search_pattern)
        
        existing_logs.sort(key=os.path.getmtime)
        
        while len(existing_logs) >= max_logs:
            oldest_log = existing_logs.pop(0) 
            os.remove(oldest_log)             

    def _print_and_log(self, text):
        print(text)
        with open(self.full_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(text + "\n")

    def calculate_shortest_path(self):
        start = self.env.start_state
        goal = self.env.end_state
        obstacles = set(self.env.obstacles)

        queue = [(start, [start])]
        visited = {start}

        while queue:
            current, path = queue.pop(0)

            if current == goal:
                return path

            x, y = current

            neighbors = [
                (x, y - 1),  # up
                (x + 1, y),  # right
                (x, y + 1),  # down
                (x - 1, y),  # left
            ]

            for next_state in neighbors:
                nx, ny = next_state

                if not (
                    0 <= nx < self.env.grid_cols
                    and 0 <= ny < self.env.grid_rows
                ):
                    continue

                if next_state in obstacles:
                    continue

                if next_state in visited:
                    continue

                visited.add(next_state)
                queue.append((next_state, path + [next_state]))

        return None
    
    def record_episode(self, success):
        self.steps_history.append(self.steps_per_ep)
        self.reward_history.append(self.rewards_per_ep)

        self.success_history.append(int(success))

        if success:
            self.successes_per_ep += 1
            self.path_lengths.append(self.steps_per_ep)

            if self.shortest_path_steps is not None:
                optimality = (
                    self.shortest_path_steps / self.steps_per_ep
                    if self.steps_per_ep > 0
                    else 0.0
                )
                self.optimality_history.append(optimality)

        self.rewards_per_ep = 0

    def get_success_rate(self):
        if not self.success_history:
            return 0.0

        return sum(self.success_history) / len(self.success_history)

    def get_average_path_length(self):
        if not self.path_lengths:
            return 0.0

        return sum(self.path_lengths) / len(self.path_lengths)

    def get_average_optimality(self):
        if not self.optimality_history:
            return 0.0

        return sum(self.optimality_history) / len(self.optimality_history)

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

    def print_learned_path(self):
        curr_state = self.env.start_state
        path = [curr_state]
        is_terminal = False
        steps = 0
        max_steps = (self.env.grid_rows * self.env.grid_cols) * 2

        was_training = self.agent.main_net.training
        original_agent_pos = self.env.agent_pos
        self.agent.main_net.eval()
        try:
            with torch.no_grad():
                while not is_terminal and steps < max_steps:
                    self.env.agent_pos = curr_state
                    state_tensor = torch.as_tensor(
                        self.env.get_state(), dtype=torch.float32
                    ).unsqueeze(0)
                    best_action = self.agent.main_net(state_tensor).argmax(dim=1).item()
                    next_state, _, is_terminal, _ = self.env.take_step(
                        curr_state, best_action
                    )
                    path.append(next_state)
                    curr_state = next_state
                    steps += 1
        finally:
            self.env.agent_pos = original_agent_pos
            self.agent.main_net.train(was_training)

        path_lines = ["", "="*40, "LEARNED GREEDY PATH", "="*40]
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

        path_lines.extend((f"", f"Steps taken: {len(path) - 1}", "="*40))

        if self.shortest_path_steps is not None and steps > 0:
            optimality = self.shortest_path_steps / steps
        else:
            optimality = 0.0

        path_lines.extend((
            "",
            f"Steps taken: {steps}",
            f"Shortest valid path: {self.shortest_path_steps}",
            f"Optimality: {optimality * 100:.2f}%",
            "="*40
        ))

        

        path_text = "\n".join(path_lines)
        self._print_and_log(path_text)

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
        success_rate = self.get_success_rate()
        average_path = self.get_average_path_length()
        average_optimality = self.get_average_optimality()

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
            f"{'Shortest valid path:':<30}| "

            f"{self.shortest_path_steps if self.shortest_path_steps is not None else 'No path'}\n"

            f"{'Success rate:':<30}| "
            f"{success_rate * 100:.2f}%\n"

            f"{'Average path length:':<30}| "
            f"{average_path:.2f}\n"

            f"{'Average optimality:':<30}| "
            f"{average_optimality * 100:.2f}%\n"
        )

        self._print_and_log(summary_text)

        self.steps_per_ep = 0
        self.rewards_per_ep = 0

    def print_total_summary(self, start_time):
        elapsed_time = time.time() - start_time

        text = (
            f"\nEQLBPW Total Runtime: {elapsed_time:.2f} seconds\n"
            f"EQLBPW Total Rewards: {self.env.tracker.rewards}"
        )

        self._print_and_log(text=text)

if __name__ == "__main__":
    tracemalloc.start()
    a = EnvironmentTracker(0, 0, 0, 0, 0, 0, 0)
    a.print_episode_summary(1, 2, 1)