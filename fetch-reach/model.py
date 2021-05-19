import torch

from torch import nn


class ActorNet(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 action_range: float,
                 zero_last: bool):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim+goal_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU())
        self.fc = nn.Linear(256, action_dim)
        self.action_range = action_range
        if zero_last:
            self.fc.weight.data.fill_(0.)
            self.fc.bias.data.fill_(0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_range * torch.tanh(self.fc(self.net(x)))


class CriticNet(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 goal_dim: int,
                 action_dim: int,
                 action_range: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim+goal_dim+action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1))
        self.action_range = action_range

    def forward(self,
                x: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, action/self.action_range], dim=-1))
