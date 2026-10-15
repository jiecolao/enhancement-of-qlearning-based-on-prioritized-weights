from .agent import Agent
from .tracker import EnvironmentTracker
import random
import numpy as np


class Environment:
    # Proposed algorithm II.b: supplies the state, reward, terminal flag,
    # collision C_i, goal G_i, and distance progress D_i for each transition.
    def __init__(
            self,
            grid,
            start_state,
            end_state,
            agent: Agent,
            episodes,
            ep_tracker,
            no_of_obstacles,
            static_obstacles,
            is_dynamic_obs=False,
            ):

        self.grid_size = grid
        self.grid_rows = grid
        self.grid_cols = grid

        self.start_state = start_state
        self.end_state = end_state

        self.agent = agent
        self.episodes = episodes
        self.ep_tracker = ep_tracker

        self.no_of_obstacles = no_of_obstacles
        self.static_obstacles = static_obstacles
        self.is_dynamic_obs = is_dynamic_obs

        self.obstacles = list()

        self.agent_pos = start_state
        self.steps = 0
        self.max_steps = self.grid_cols * self.grid_cols 

        self.tracker = EnvironmentTracker(
            agent=agent,
            env=self
        )

    def reset(self):
        # Proposed algorithm II.a: initialize the episode at the configured
        # starting state and return its encoded observation.
        self.agent_pos = self.start_state
        self.steps = 0

        return self.get_state()

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

            if (
                rand_state != self.start_state
                and rand_state != self.end_state
                and rand_state not in self.obstacles
            ):
                self.obstacles.append(rand_state)
                dynamic_added += 1

    def take_step(self, state, action):
        # Proposed algorithm II.b: execute a discrete grid action and produce
        # the transition fields stored by the EQLBPW replay buffer.
        x, y = state

        # Actions:
        # 0 = up
        # 1 = right
        # 2 = down
        # 3 = left

        if action == 0:
            y = max(0, y - 1)

        elif action == 1:
            x = min(self.grid_cols - 1, x + 1)

        elif action == 2:
            y = min(self.grid_rows - 1, y + 1)

        elif action == 3:
            x = max(0, x - 1)

        attempted_state = (x, y)

        collision = attempted_state in self.obstacles

        if collision:
            next_state = state
        else:
            next_state = attempted_state

        # check whether the agent reached the goal
        goal_reached = next_state == self.end_state

        # Proposed 8: Manhattan distance is used to measure path progress.
        old_distance = self.distance_to_goal(state)
        new_distance = self.distance_to_goal(next_state)

        distance_progress = old_distance - new_distance

        # Proposed 7: shaped rewards distinguish collision, goal, and ordinary
        # movement; these values are experiment semantics, not replay priority.
        if collision:
            reward = -10.0

        elif goal_reached:
            reward = 10.0

        else:
            reward = -0.1

        # Proposed algorithm II.b: goal arrival is terminal. The simulator also
        # enforces the maximum-step episode boundary.
        is_terminal = goal_reached

        # Update environment state
        self.agent_pos = next_state
        self.steps += 1

        info = {
            "collision": collision,
            "goal": goal_reached,
            "distance_progress": distance_progress,
        }

        return next_state, reward, is_terminal, info

    def get_state(self):
        # Proposed algorithm: state representation presented to the DQN. It
        # combines normalized agent/goal coordinates with a local 5x5 obstacle
        # window (4 + 25 = 29 features).
        agent_x, agent_y = self.agent_pos
        goal_x, goal_y = self.end_state

        state = [
            agent_x / (self.grid_rows - 1),
            agent_y / (self.grid_cols - 1),
            goal_x / (self.grid_rows - 1),
            goal_y / (self.grid_cols - 1),
        ]

        # Add 5x5 local obstacle information
        for dx in range(-2, 3):
            for dy in range(-2, 3):

                x = agent_x + dx
                y = agent_y + dy

                # Treat outside the grid as obstacles
                if (
                    x < 0
                    or x >= self.grid_rows
                    or y < 0
                    or y >= self.grid_cols
                ):
                    state.append(1.0)

                elif (x, y) in self.obstacles:
                    state.append(1.0)

                else:
                    state.append(0.0)

        return np.array(state, dtype=np.float32)

    def distance_to_goal(self, state):
        # Proposed 8: Manhattan distance |x - x_goal| + |y - y_goal|.
        return (
            abs(state[0] - self.end_state[0])
            + abs(state[1] - self.end_state[1])
        )

    def _test(self):
        print("environment.py accessed!")