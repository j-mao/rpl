from gym.envs.registration import register

from robot_env.pendulum import InvertedDoublePendulumEnv


register(id='InvertedDoublePendulumTarget-v0',
         entry_point='robot_env.pendulum:InvertedDoublePendulumEnv',
         max_episode_steps=1000,
         reward_threshold=9100.0)
