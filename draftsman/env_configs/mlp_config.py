from typing import Optional, Tuple

import gym
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

import numpy as np
import torch
import torch.nn as nn

from draftsman.env_configs.schedule import ConstantSchedule
import draftsman.infrastructure.pytorch_util as ptu


def basic_mlp_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 1e-3,
    total_steps: int = 1000000,
    discount: float = 0.95,
    clip_grad_norm: Optional[float] = None,
    batch_size: int = 128,
    **kwargs,
):
    def make_model(observation_shape: Tuple[int, ...], output_size: int) -> nn.Module:
        return ptu.build_mlp(
            input_size=np.prod(observation_shape),
            output_size=output_size,
            n_layers=num_layers,
            size=hidden_size,
        )

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def make_env():
        return RecordEpisodeStatistics(gym.make(env_name), 100)

    log_string = "{}_{}_s{}_l{}_d{}".format(
        exp_name or "mlp",
        env_name,
        hidden_size,
        num_layers,
        discount,
    )

    return {
        "agent_kwargs": {
            "make_model": make_model,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "clip_grad_norm": clip_grad_norm,
        },
        "log_name": log_string,
        "make_env": make_env,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "env_name": env_name,
        "agent": "mlp",
        **kwargs,
    }
