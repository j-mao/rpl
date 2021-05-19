import numpy as np

from dataclasses import dataclass
from easyrl.envs.vec_env import VecEnv
from easyrl.utils.rl_logger import TensorboardLogger

from agent import DDPGAgent
from config import ExperimentConfig
from replay import ReplayBuffer, TrajectorySet


@dataclass
class DDPGEngine:
    cfg: ExperimentConfig
    agent: DDPGAgent
    train_env: VecEnv
    eval_env: VecEnv
    replay: ReplayBuffer
    tf_logger: TensorboardLogger

    step = 0

    def train(self) -> None:
        for epoch in range(self.cfg.num_epochs):
            for cycle in range(self.cfg.cycles_per_epoch):
                print(f'Epoch {epoch} / Cycle {cycle}')
                for traj in self._rollout(train=True, log_tag='train'):
                    self.replay.append(traj)
                self.agent.normalizer.update(
                    self.replay.sample(self.cfg.batch_size))

                for batch in range(self.cfg.batches_per_cycle):
                    trajs = self.replay.sample(self.cfg.batch_size)
                    critic_only = (epoch < self.cfg.burn_epochs)
                    log = self.agent.optimize(trajs, critic_only=critic_only)
                    self.tf_logger.save_dict({'scalar': log}, step=self.step)
                self.agent.soft_update_targets()

            self._rollout(train=False, log_tag='eval')

    def _rollout(self,
                 train: bool,
                 log_tag: str) -> TrajectorySet:
        env = self.train_env if train else self.eval_env
        num_trajs = self.cfg.episodes_per_cycle \
                    if train                    \
                    else self.cfg.num_eval_envs
        trajs = TrajectorySet.empty(self.cfg, num_trajs)
        rewards = np.zeros(num_trajs)
        success = np.zeros(num_trajs)
        obs = env.reset()
        for i in range(self.cfg.episode_steps):
            action = self.agent.get_action(obs, sample=train)
            next_obs, r, _, infos = env.step(action)
            trajs[:,i] = (obs['observation'],
                          next_obs['observation'],
                          action,
                          next_obs['desired_goal'],
                          next_obs['achieved_goal'])
            obs = next_obs
            rewards += r
            success = np.array([info['is_success'] for info in infos])
        if train:
            self.step += num_trajs * self.cfg.episode_steps
        log = {f'{log_tag}/reward': rewards.mean(),
               f'{log_tag}/success': success.mean()}
        self.tf_logger.save_dict({'scalar': log}, step=self.step)
        return trajs
