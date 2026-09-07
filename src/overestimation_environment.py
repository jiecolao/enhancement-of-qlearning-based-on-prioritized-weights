import numpy as np

from EQLBPW.environment import Environment as EQLBPWEnvironment
from QLBPW.environment import Environment as QLBPWEnvironment


class StochasticOverestimationQLBPWEnvironment(QLBPWEnvironment):
    """Production QLBPW environment with zero-mean reward noise.

    The movement dynamics and state representation remain those of the
    production environment. Only the non-terminal reward is made stochastic.
    """

    def __init__(self, *args, reward_std=2.0, rng=None, **kwargs):
        self.reward_std = float(reward_std)
        self.rng = rng if rng is not None else np.random.default_rng()
        super().__init__(*args, **kwargs)

    def take_step(self, state, action):
        next_state, reward, terminal = super().take_step(state, action)
        if not terminal:
            reward = float(reward) + float(self.rng.normal(0.0, self.reward_std))
        return next_state, reward, terminal


class StochasticOverestimationEQLBPWEnvironment(EQLBPWEnvironment):
    """Production EQLBPW environment with the same stochastic reward process.

    The benchmark uses the same normalized reward scale as QLBPW:
    collision=-1, goal=+1, ordinary movement=0, plus zero-mean Gaussian noise
    on non-terminal transitions.
    """

    def __init__(self, *args, reward_std=2.0, rng=None, **kwargs):
        self.reward_std = float(reward_std)
        self.rng = rng if rng is not None else np.random.default_rng()
        super().__init__(*args, **kwargs)

    def take_step(self, state, action):
        next_state, _, terminal, info = super().take_step(state, action)
        collision = bool(info.get("collision", False))
        goal = bool(info.get("goal", False))

        if collision:
            reward = -1.0
        elif goal:
            reward = 1.0
        else:
            reward = 0.0 + float(self.rng.normal(0.0, self.reward_std))

        return next_state, reward, terminal, info
