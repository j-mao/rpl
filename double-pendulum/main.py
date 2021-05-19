import sys
sys.path.append('easyrl')

from pathlib import Path

import torch
from torch import nn
from torch.distributions import Independent, Normal

from easyrl.configs import cfg, set_config
from easyrl.agents.ppo_agent import PPOAgent
from easyrl.engine.ppo_engine import PPOEngine
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy
from easyrl.models.mlp import MLP
from easyrl.models.value_net import ValueNet
from easyrl.runner.nstep_runner import EpisodicRunner
from easyrl.utils.common import set_random_seed
from easyrl.utils.gym_util import make_vec_env

import pybullet_envs
import robot_env

def build_mlp(observation_size):
    return MLP(
        input_size=observation_size,
        hidden_sizes=[64, 64],
        output_size=64,
        hidden_act=nn.Tanh,
        output_act=nn.Tanh,
    )

class ResidualActorPolicy(nn.Module):
    def __init__(self, obs, act, pretrain):
        super().__init__()
        self.mlp = build_mlp(obs)
        self.fc = nn.Linear(64, act)
        self.fc.weight.data.fill_(0.)
        self.fc.bias.data.fill_(0.)
        self.pretrain = pretrain
        self.logstd = nn.Parameter(torch.full((act,), -0.51))

    def forward(self, x):
        with torch.no_grad():
            u = self.pretrain.actor(x)[0].mean
        body_x = self.mlp(x)
        r = u + self.fc(body_x)
        return Independent(Normal(loc=r, scale=torch.exp(self.logstd)), 1), body_x


def train(train_env_id, eval_env_id, logdir, pretrain, seed):
    set_config('ppo')
    cfg.alg.num_envs = 16
    cfg.alg.episode_steps = 1000
    cfg.alg.log_interval = 1
    cfg.alg.eval_interval = 5
    cfg.alg.env_name = train_env_id
    cfg.alg.device = 'cpu'
    cfg.alg.seed = seed
    cfg.alg.save_dir = Path.cwd().absolute().joinpath('logs', logdir).as_posix()
    set_random_seed(cfg.alg.seed)

    env = make_vec_env(cfg.alg.env_name, cfg.alg.num_envs, seed=cfg.alg.seed)
    env.reset()
    eval_env = make_vec_env(eval_env_id, 1, seed=cfg.alg.seed)

    if pretrain is None:
        actor = DiagGaussianPolicy(
            build_mlp(env.observation_space.shape[0]),
            in_features=64,
            action_dim=env.action_space.shape[0])
    else:
        actor = ResidualActorPolicy(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            pretrain)
    critic = ValueNet(
        build_mlp(env.observation_space.shape[0]),
        in_features=64,
    )

    agent = PPOAgent(actor=actor, critic=critic, env=env)
    runner = EpisodicRunner(agent=agent, env=env, eval_env=eval_env)
    engine = PPOEngine(agent=agent, runner=runner)

    if pretrain is not None:
        cfg.alg.max_steps = 200000
        lr, cfg.alg.policy_lr = cfg.alg.policy_lr, 0
        engine.train()
        cfg.alg.policy_lr = lr

    cfg.alg.max_steps = 1000000
    engine.train()

    return agent

def main(seed):
    source_env_id = 'InvertedDoublePendulumBulletEnv-v0'
    target_env_id = 'InvertedDoublePendulumTarget-v0'
    agent_source = train(source_env_id, target_env_id, 'source', pretrain=None, seed=seed)
    agent_residual = train(target_env_id, target_env_id, 'residual', pretrain=agent_source, seed=seed)
    agent_target = train(target_env_id, target_env_id, 'target', pretrain=None, seed=seed)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main(int(sys.argv[1]))
