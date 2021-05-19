import numpy as np

from gym.envs import mujoco


class InvertedDoublePendulumEnv(mujoco.InvertedDoublePendulumEnv):
    """Adapts the MuJoCo inverted double-pendulum environment to have the same
    observation space as the PyBullet environment.
    """

    def _get_obs(self) -> np.ndarray:
        x, theta, gamma = self.sim.data.qpos
        vx, theta_dot, gamma_dot = self.sim.data.qvel
        pos_x, _, _ = self.sim.data.site_xpos[0]
        return np.array([x, vx, pos_x,
                         np.cos(theta), np.sin(theta), theta_dot,
                         np.cos(gamma), np.sin(gamma), gamma_dot])
