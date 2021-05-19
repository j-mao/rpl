import numpy as np
import torch

from copy import deepcopy
from dataclasses import dataclass
from easyrl.utils.torch_util import torch_float
from torch import nn, optim
from torch.nn import functional as F
from typing import Callable, Dict, Optional

from config import ExperimentConfig
from normalize import Normalizer
from replay import TrajectorySet


@dataclass
class DDPGAgent:
    cfg: ExperimentConfig
    actor: nn.Module
    critic: nn.Module
    normalizer: Normalizer
    reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    pretrain: Optional[nn.Module]

    def __post_init__(self):
        self.actor_targ = deepcopy(self.actor)
        self.critic_targ = deepcopy(self.critic)
        self.actor_optim = optim.Adam(self.actor.parameters(),
                                      lr=self.cfg.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(),
                                       lr=self.cfg.critic_lr)

    @torch.no_grad()
    def get_action(self,
                   obs: Dict[str, np.ndarray],
                   sample: bool) -> np.ndarray:
        state = np.concatenate([obs['observation'], obs['desired_goal']],
                               axis=-1)
        state_tensor = torch_float(self.normalizer(state))
        u = self.actor(state_tensor).numpy()
        if sample:
            noise_scale = self.cfg.noise_eps * self.cfg.action_range
            u += noise_scale * np.random.randn(*u.shape)
            u = np.clip(u, -self.cfg.action_range, self.cfg.action_range)
            u_rand = np.random.uniform(low=-self.cfg.action_range,
                                       high=self.cfg.action_range,
                                       size=u.shape)
            use_rand = np.random.binomial(1, self.cfg.epsilon, size=u.shape[0])
            u += use_rand.reshape(-1, 1) * (u_rand - u)
        if self.pretrain is not None:
            u += self.pretrain(state_tensor).numpy()
        return u

    def optimize(self,
                 trajs: TrajectorySet,
                 critic_only: bool) -> Dict[str, float]:
        obs, next_obs, actions, desired_goal, achieved_goal = trajs.data
        state = torch_float(self.normalizer(
            np.concatenate([obs, desired_goal], axis=-1)))
        next_state = torch_float(self.normalizer(
            np.concatenate([next_obs, desired_goal], axis=-1)))
        u = torch_float(actions)
        r = torch_float(self.reward_fn(achieved_goal, desired_goal, info=None))

        with torch.no_grad():
            q_next = self.critic_targ(next_state, self.actor_targ(next_state))
            y = r.view(-1, 1) + self.cfg.discount * q_next
            y = torch.clamp(y, -1 / (1 - self.cfg.discount), 0)
        q = self.critic(state, u)
        critic_loss = F.mse_loss(q, y)

        if not critic_only:
            u_pred = self.actor(state)
            actor_loss = -self.critic(state, u_pred).mean()
            actor_reg = torch.square(u_pred / self.cfg.action_range).mean()
            actor_loss += self.cfg.action_reg * actor_reg

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
        else:
            actor_loss = torch.tensor(0)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        return {'train/actor_loss': actor_loss.item(),
                'train/critic_loss': critic_loss.item()}

    def soft_update_targets(self) -> None:
        self._soft_update(self.actor_targ, self.actor)
        self._soft_update(self.critic_targ, self.critic)

    def _soft_update(self, targ_net: nn.Module, net: nn.Module) -> None:
        for targ_param, param in zip(targ_net.parameters(), net.parameters()):
            targ_param.data.copy_((1-self.cfg.tau) * targ_param.data +
                                  self.cfg.tau * param.data)
