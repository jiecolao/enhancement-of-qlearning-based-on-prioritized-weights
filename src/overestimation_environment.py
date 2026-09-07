import numpy as np

from EQLBPW.environment import Environment as EQLBPWEnvironment
from QLBPW.environment import Environment as QLBPWEnvironment


class _RewardMatrixMixin:
    """Add reproducible, episode/action-indexed reward noise to an environment."""

    def _init_stochastic_rewards(self, reward_std, rng, reward_matrix):
        self.reward_std = float(reward_std)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reward_matrix = None if reward_matrix is None else np.asarray(reward_matrix, dtype=float)
        self.current_episode = 0

        if self.reward_matrix is not None and self.reward_matrix.ndim != 2:
            raise ValueError("reward_matrix must be a 2-D array: [episode, action]")
        if self.reward_matrix is not None and self.reward_matrix.shape[1] < 4:
            raise ValueError("reward_matrix must contain at least four action columns")

    def start_episode(self, episode_index):
        self.current_episode = int(episode_index)
        if self.reward_matrix is not None and not 0 <= self.current_episode < len(self.reward_matrix):
            raise IndexError("episode_index is outside reward_matrix")

    def _noise(self, action):
        if self.reward_matrix is not None:
            return float(self.reward_matrix[self.current_episode, int(action)])
        return float(self.rng.normal(0.0, self.reward_std))


class StochasticOverestimationQLBPWEnvironment(_RewardMatrixMixin, QLBPWEnvironment):
    """Production QLBPW environment with reproducible stochastic rewards."""

    def __init__(self, *args, reward_std=2.0, rng=None, reward_matrix=None, **kwargs):
        self._init_stochastic_rewards(reward_std, rng, reward_matrix)
        super().__init__(*args, **kwargs)

    def take_step(self, state, action):
        next_state, reward, terminal = super().take_step(state, action)
        if not terminal:
            reward = float(reward) + self._noise(action)
        return next_state, reward, terminal


class StochasticOverestimationEQLBPWEnvironment(_RewardMatrixMixin, EQLBPWEnvironment):
    """Production EQLBPW environment using the same reward process as QLBPW."""

    def __init__(self, *args, reward_std=2.0, rng=None, reward_matrix=None, **kwargs):
        self._init_stochastic_rewards(reward_std, rng, reward_matrix)
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
            reward = self._noise(action)

        return next_state, reward, terminal, info
