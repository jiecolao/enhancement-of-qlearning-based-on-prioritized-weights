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
        self.obstacle_encountered = 0
        self.goal_count = 0
        self.steps_per_ep = 0
        self.rewards_per_ep = 0

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

    def print_episode_summary(self):
        pass

class AgentTracker:

    def __init__(self):
        pass