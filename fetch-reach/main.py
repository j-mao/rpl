import sys
sys.path.append('easyrl')

import gym
import os
import torch

from easyrl.utils.gym_util import make_vec_env
from easyrl.utils.rl_logger import TensorboardLogger
from typing import List, Optional

from agent import DDPGAgent
from config import ExperimentConfig
from engine import DDPGEngine
from model import ActorNet, CriticNet
from normalize import Normalizer
from replay import HERReplayBuffer
import robot_env


def set_env_metadata(env_id: str, cfg: ExperimentConfig) -> gym.Env:
    env = gym.make(env_id)
    cfg.obs_dim = env.observation_space['observation'].shape[0]
    cfg.goal_dim = env.observation_space['desired_goal'].shape[0]
    cfg.action_dim = env.action_space.shape[0]
    cfg.action_range = env.action_space.high[0]
    return env

def train(train_env_id: str,
          eval_env_id: str,
          logdir: str,
          cfg: ExperimentConfig,
          save_path: str,
          pretrain_path: Optional[str] = None) -> DDPGAgent:
    pretrain = torch.load(os.path.join(pretrain_path)) \
               if pretrain_path is not None            \
               else None
    env = set_env_metadata(train_env_id, cfg)
    train_env = make_vec_env(train_env_id,
                             num_envs=cfg.episodes_per_cycle,
                             no_timeout=True,
                             seed=cfg.seed)
    eval_env = make_vec_env(eval_env_id,
                            num_envs=cfg.num_eval_envs,
                            no_timeout=True,
                            seed=cfg.seed+100)
    replay = HERReplayBuffer(cfg=cfg)
    tf_logger = TensorboardLogger(logdir)
    actor = ActorNet(obs_dim=cfg.obs_dim,
                     goal_dim=cfg.goal_dim,
                     action_dim=cfg.action_dim,
                     action_range=cfg.action_range,
                     zero_last=(pretrain_path is not None))
    critic = CriticNet(obs_dim=cfg.obs_dim,
                       goal_dim=cfg.goal_dim,
                       action_dim=cfg.action_dim,
                       action_range=cfg.action_range)
    normalizer = Normalizer(cfg.obs_dim+cfg.goal_dim) \
                 if pretrain is None                  \
                 else pretrain.normalizer
    agent = DDPGAgent(cfg=cfg,
                      actor=actor,
                      critic=critic,
                      normalizer=normalizer,
                      reward_fn=env.compute_reward,
                      pretrain=getattr(pretrain, 'actor', None))
    engine = DDPGEngine(cfg=cfg,
                        agent=agent,
                        train_env=train_env,
                        eval_env=eval_env,
                        replay=replay,
                        tf_logger=tf_logger)
    engine.train()

    env.close()
    train_env.close()
    eval_env.close()
    torch.save(agent, os.path.join(save_path))
    return agent

def save_path(logdir: str, mode: str) -> str:
    return os.path.join('output', 'model', logdir, f'{mode}.pth')

def main(cli_opts: List[str], **kwargs) -> None:
    assert len(cli_opts) == 4
    logdir, mode, source_env_id, target_env_id = cli_opts
    options = {
        'source': dict(train_env_id=source_env_id,
                       cfg=ExperimentConfig(**kwargs)),
        'residual': dict(train_env_id=target_env_id,
                         cfg=ExperimentConfig(burn_epochs=5, **kwargs),
                         pretrain_path=save_path(logdir, 'source')),
        'target': dict(train_env_id=target_env_id,
                       cfg=ExperimentConfig(**kwargs)),
    }
    train(eval_env_id=target_env_id,
          logdir=os.path.join('output', 'logs', logdir, mode),
          save_path=save_path(logdir, mode),
          **options[mode])


if __name__ == '__main__':
    # Usage: python3.7 main.py logdir mode source_env_id target_env_id seed
    torch.set_num_threads(4)
    seed = int(sys.argv[-1])
    main(sys.argv[1:-1], seed=seed)
