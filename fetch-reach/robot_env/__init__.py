from gym.envs.registration import register

from robot_env.slippery import SlipperyFetchPushEnv


register(id='SlipperyFetchPush-v1',
         entry_point='robot_env.slippery:SlipperyFetchPushEnv',
         max_episode_steps=50)
