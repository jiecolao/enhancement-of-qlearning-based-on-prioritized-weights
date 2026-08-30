
class EnvironmentTracker:

    def __init__(self):
        self.pos = 0
        self.steps = 0
        self.obstacle_encountered = 0
        self.goal_count = 0
        self.steps_per_ep = 0
        self.rewards_per_ep = 0
        

class AgentTracker:

    def __init__(self):
        pass