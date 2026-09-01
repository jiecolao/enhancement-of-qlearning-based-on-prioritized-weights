from EQLBPW.agent import Agent
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

    def _reset(self):
        self.agent_pos = self.start_state
        self.steps = 0
        return

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

    def take_step(self, state, action):
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

        is_terminal = False
        
        if next_state in self.obstacles:
            reward = -1
            next_state = state
        elif next_state == self.end_state:
            reward = 1
            is_terminal = True
        else:
            reward = 0
        return next_state, reward, is_terminal

    def _test(self):
        print("environment.py accessed!")