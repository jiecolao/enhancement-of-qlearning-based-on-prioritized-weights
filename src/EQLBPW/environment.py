from EQLBPW.agent import Agent

class Environment:

    def __init__(
            self,
            grid,
            obstacles,
            agent: Agent,
            ):
        self.grid = grid
        self.obstacles = obstacles
        self.episodes = agent.episodes

        self.agent = agent

    def simulate(self):

        # I. Initialization

        for e in self.episodes:
            # 1. Initialize starting status s
            # 2. Repeat until s is terminated
            pass

    def _generate_obstacles(self):
        pass

    def _move(self):
        pass