from EQLBPW.agent import Agent
from utility import EnvironmentTracker

class Environment:
    def __init__(
            self,
            grid,
            start_state,
            end_state,
            agent: Agent,
            episodes,
            no_of_obstacles,
            static_obstacles,
            is_dynamic_obs = False,
            ):
        
        self.grid_size = grid
        self.grid_rows = grid
        self.grid_cols = grid
        self.start_state = start_state
        self.end_state = end_state

        self.agent = agent
        self.episodes = episodes

        self.no_of_obstacles = no_of_obstacles
        self.static_obstacles = static_obstacles
        self.is_dynamic_obs = is_dynamic_obs

        self.obstacles = list()

        self.agent_pos = start_state
        self.steps = 0
        self.max_steps = self.grid_cols * self.grid_cols ** 2 # 50
        self.tracker = EnvironmentTracker()

    def simulate(self):
        
        for e in range(self.episodes):
            curr_state = self.start_state
            is_terminal = False

            while not is_terminal and self.steps <= self.max_steps:
                action = agent._e_greedy(curr_state)
                next_state, reward, is_terminal = self._take_step(curr_state, action)

    def _reset(self):
        self.agent_pos = self.start_state
        self.steps = 0
        return

    def _generate_obstacles(self):
        # Reset the obstacles list to just the static ones
        self.obstacles.clear()
        # self.obstacles = self.static_obstacles.copy()
        self.obstacles = list(self.static_obstacles)

        if not self.dynamic_obs_enabled:
            return

        dynamic_added = 0
        while dynamic_added < self.num_dynamic_obs:
            # Pick a random coordinate on the grid (x, y)
            rand_x = random.randint(0, self.grid_cols - 1)
            rand_y = random.randint(0, self.grid_rows - 1)
            rand_state = (rand_x, rand_y)
            
            # Make sure it's not the start, goal, or already an obstacle
            if (rand_state != self.start_state and 
                rand_state != self.goal_state and 
                rand_state not in self.obstacles):
                
                self.obstacles.append(rand_state)
                dynamic_added += 1

    def _take_step(self, state, action):
        self.steps += 1 # increment when obstacle is encountered?

        # State is a (x, y) coordinate tuple where x is column, y is row
        x, y = state

        # Actions: 0="up", 1="right", 2="down", 3="left"
        if action == 0:
            y = max(0, y - 1)
        elif action == 1:
            x = min(self.grid_cols - 1, x + 1)
        elif action == 2:
            y = min(self.grid_rows - 1, y + 1)
        elif action == 3:
            x = max(0, x - 1)

        next_state = (x, y)

        # Calculate Reward and Terminal Status
        is_terminal = False
        
        # Check if next state is an obstacle
        if next_state in self.obstacles:
            reward = -1
            self.obstaclesCount += 1
            next_state = state
        elif next_state == self.end_state:
            reward = 1
            is_terminal = True
            self.tracker.goal_count += 1
        else:
            reward = 0
        return next_state, reward, is_terminal

    def _test(self):
        print("environment.py accessed!")

if __name__ == "__main__":
    BASE_OBSTACLES = {
                (1, 0),                 (4, 0),                             (8, 0),
                                                        (6, 1),
        (0, 2),                 (3, 2),
                        (2, 3),                 (5, 3),         (7, 3),     (8, 3), 
        (0, 4),                 (3, 4),
                                                (5, 5), (6, 5), (7, 5), 
                (1, 6),                         (5, 6),         (7, 6), 
                                (3, 7),         (5, 7),         (7, 7),
        (0, 8)
    }

    agent = Agent(
        alpha=0.1,
        gamma=0.9,
        beta=0.3,
        e=0.9, 
        e_min=0.1, 
        e_decay=0.01,        
        no_of_states=4, 
        no_of_actions=4,
        batch_size=5, 
        max_buffer=1000,
    )

    env = Environment(
            grid = 16,
            start_state = (0, 0),
            end_state = (1, 1),
            agent = agent,
            episodes = 100,
            no_of_obstacles = 5,
            static_obstacles = BASE_OBSTACLES,
            is_dynamic_obs = False,
    )
    env.simulate()