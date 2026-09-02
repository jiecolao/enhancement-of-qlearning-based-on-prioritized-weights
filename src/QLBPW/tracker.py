from datetime import datetime
import tracemalloc
import os
import glob

class EnvironmentTracker:

    def __init__(self, grid, rows, cols, start, end, obstacles):
        self.grid = grid
        self.rows = rows
        self.cols = cols
        self.start_state = start
        self.end_state = end
        self.obstacles = obstacles

        self.pos = 0
        self.steps = 0
        self.rewards = 0
        self.pos_rewards = 0
        self.neg_rewards = 0
        self.obstacle_encountered = 0
        self.goal_count = 0
        self.steps_per_ep = 0
        self.rewards_per_ep = 0

        self.shortest_recorded_steps = 0
        self.path_per_ep = []
        
        self.log_folder = "src/QLBPW/training_logs"
        os.makedirs(self.log_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"log_{timestamp}.txt"
        self.full_log_path = os.path.join(self.log_folder, self.log_filename)
        self._manage_log_limit(max_logs=10)

    def _manage_log_limit(self, max_logs):
        search_pattern = os.path.join(self.log_folder, "log_*.txt")
        existing_logs = glob.glob(search_pattern)
        
        existing_logs.sort(key=os.path.getmtime)
        
        while len(existing_logs) >= max_logs:
            oldest_log = existing_logs.pop(0) 
            os.remove(oldest_log)             

    def print_live_grid(self, agent_pos):
            print("\n" + "="*40)
            print("ENVIRONMENT")
            print("="*40)
            for y in range(self.rows):
                row_str = ""
                for x in range(self.cols):
                    state = (x, y)
                    if state == agent_pos:
                        row_str += " 🤖 \t"
                    elif state == self.start_state:
                        row_str += " 🚪 \t"
                    elif state == self.end_state:
                        row_str += " 🏁 \t"
                    elif state in self.obstacles:
                        row_str += " 🧱 \t"
                    else:
                        row_str += " . \t"
                print(row_str)
            print("="*40)

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
            f"{'Episode Completion Time:':<30}| {elapsed:.2f} seconds\n"
            f"{'Memory usage:':<30}| Current: {current / (1024 * 1024):.2f} MB, Peak: {peak / (1024 * 1024):.2f} MB\n"
            f"{'Total Steps:':<30}| {self.steps}\n"
            f"{'Total Obstacles Encountered:':<30}| {self.obstacle_encountered}\n"
            f"{'Total Goals:':<30}| {self.goal_count}\n"
            f"{'Total Rewards:':<30}| {self.rewards}\n"
            f"{' ├── Positive Rewards:':<30}| {self.pos_rewards}\n"
            f"{' └── Negative Rewards:':<30}| {self.neg_rewards}\n"
        )

        print(summary_text)

        with open(self.full_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(summary_text + "\n")

        self.steps_per_ep = 0
        self.rewards_per_ep = 0

class AgentTracker:

    def __init__(self):
        pass

if __name__ == "__main__":
    tracemalloc.start()
    a = EnvironmentTracker(0, 0, 0, 0, 0, 0, 0)
    a.print_episode_summary(1, 2, 1)