import gym

from gym.envs.robotics import FetchPushEnv


def slippery_mujoco_env(cls):
    def replace_init(old_init):
        def thunk(self, *args, **kwargs):
            old_init(self, *args, **kwargs)
            for i in range(len(self.sim.model.geom_friction)):
                self.sim.model.geom_friction[i, 0] *= 0.18
        return thunk

    cls.__init__ = replace_init(cls.__init__)
    return cls


@slippery_mujoco_env
class SlipperyFetchPushEnv(FetchPushEnv):
    pass
