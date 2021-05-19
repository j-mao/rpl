from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    num_epochs: int = 25
    burn_epochs: int = 0
    cycles_per_epoch: int = 50
    episodes_per_cycle: int = 16
    episode_steps: int = 50

    num_eval_envs: int = 8
    seed: int = 1

    batches_per_cycle: int = 100
    batch_size: int = 256

    replay_capacity: int = 1000000
    her_future_prob: float = 0.8

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    discount: float = 0.98
    tau: float = 0.05
    noise_eps: float = 0.2
    epsilon: float = 0.3
    action_reg: float = 1.0
