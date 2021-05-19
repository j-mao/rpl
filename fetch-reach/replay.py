import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union

from config import ExperimentConfig


@dataclass
class TrajectorySet:
    obs: np.ndarray
    next_obs: np.ndarray
    actions: np.ndarray
    desired_goal: np.ndarray
    achieved_goal: np.ndarray

    @property
    def data(self) -> Tuple[np.ndarray, ...]:
        return (self.obs,
                self.next_obs,
                self.actions,
                self.desired_goal,
                self.achieved_goal)

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: Union[int, slice, tuple]) -> 'TrajectorySet':
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = idx + (...,)
        return TrajectorySet(obs=self.obs[idx],
                             next_obs=self.next_obs[idx],
                             actions=self.actions[idx],
                             desired_goal=self.desired_goal[idx],
                             achieved_goal=self.achieved_goal[idx])

    def __setitem__(self,
                    idx: Union[int, slice, tuple],
                    value: Tuple[np.ndarray, ...]):
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx = idx + (...,)
        (self.obs[idx],
         self.next_obs[idx],
         self.actions[idx],
         self.desired_goal[idx],
         self.achieved_goal[idx]) = value

    @classmethod
    def empty(cls, cfg: ExperimentConfig, num_trajs: int) -> 'TrajectorySet':
        shape = (num_trajs, cfg.episode_steps)
        return cls(obs=np.empty(shape + (cfg.obs_dim,)),
                   next_obs=np.empty(shape + (cfg.obs_dim,)),
                   actions=np.empty(shape + (cfg.action_dim,)),
                   desired_goal=np.empty(shape + (cfg.goal_dim,)),
                   achieved_goal=np.empty(shape + (cfg.goal_dim,)))


@dataclass
class ReplayBuffer(ABC):
    cfg: ExperimentConfig

    def __post_init__(self):
        self.size = 0
        self.buf = TrajectorySet.empty(self.cfg, self.cfg.replay_capacity)

    def append(self, traj: TrajectorySet) -> None:
        idx = np.random.randint(self.cfg.replay_capacity)
        if self.size < self.cfg.replay_capacity:
            idx = self.size
            self.size += 1
        self.buf[idx] = traj.data

    @abstractmethod
    def sample(self, batch_size: int) -> TrajectorySet:
        pass


class HERReplayBuffer(ReplayBuffer):
    def sample(self, batch_size: int) -> TrajectorySet:
        T = self.cfg.episode_steps
        traj_idx = np.random.randint(self.size, size=batch_size)
        step_idx = np.random.randint(T, size=batch_size)
        result = self.buf[traj_idx, step_idx]
        her_idx = np.where(np.random.uniform(size=batch_size) <
                           self.cfg.her_future_prob)
        goal_idx = np.random.randint(step_idx[her_idx], T)
        goal_steps = self.buf[traj_idx[her_idx], goal_idx]
        result.desired_goal[her_idx] = goal_steps.achieved_goal
        return result
