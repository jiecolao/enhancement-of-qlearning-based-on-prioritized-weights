from QLBPW.agent import Agent
from tracker import EnvironmentTracker
import random

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
        self.max_steps = self.grid_cols * self.grid_cols ** 2
        self.tracker = EnvironmentTracker(
            grid=grid,
            rows=grid,
            cols=grid,
            start=start_state,
            end=end_state,
            obstacles=static_obstacles
        )

    def generate_obstacles(self):
        self.obstacles.clear()
        self.obstacles = list(self.static_obstacles)

        if not self.is_dynamic_obs:
            return

        dynamic_added = 0
        while dynamic_added < self.no_of_obstacles:
            rand_x = random.randint(0, self.grid_cols - 1)
            rand_y = random.randint(0, self.grid_rows - 1)
            rand_state = (rand_x, rand_y)
            
            if (rand_state != self.start_state and 
                rand_state != self.end_state and 
                rand_state not in self.obstacles):
                
                self.obstacles.append(rand_state)
                dynamic_added += 1
