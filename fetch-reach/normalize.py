import numpy as np

from typing import Tuple

from replay import TrajectorySet


class Normalizer:
    eps = 0.01

    def __init__(self, shape: Tuple[int, ...]):
        self.sum1 = np.zeros(shape)
        self.sum2 = np.zeros(shape)
        self.mean = np.zeros(shape)
        self.std = np.ones(shape)
        self.n = 1

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.std, -5, 5)

    def update(self, trajs: TrajectorySet) -> np.ndarray:
        x = np.concatenate([trajs.obs, trajs.desired_goal], axis=-1)
        x = x.reshape(-1, x.shape[-1])
        self.sum1 += x.sum(axis=0)
        self.sum2 += (x*x).sum(axis=0)
        self.n += x.shape[0]
        self.mean = self.sum1 / self.n
        self.std = np.sqrt(np.maximum(self.eps*self.eps,
                                      self.sum2/self.n - self.mean*self.mean))
