from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn

import math

import numpy as np

import draftsman.infrastructure.pytorch_util as ptu


class MLPAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        output_size: int,
        make_model: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.model = make_model(observation_shape, output_size)
        self.optimizer = make_optimizer(self.model.parameters())
        self.lr_scheduler = make_lr_schedule(self.optimizer)

        self.observation_shape = observation_shape
        self.output_size = output_size
        self.clip_grad_norm = clip_grad_norm

        self.loss_fn = nn.CrossEntropyLoss() # TODO: test mse vs cross entropy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model(observation)

    def compute_loss(
        self,
        obs: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        predictions = self.forward(obs)
        loss = self.loss_fn(predictions, target)
        return loss

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the MLP agent.
        """

        accepted_tokens = ((action + reward) // 2).type(torch.LongTensor).to(ptu.device)
        # print(obs, accepted_tokens)
        # print(obs.shape, accepted_tokens.shape)
        loss = self.compute_loss(obs, accepted_tokens)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        self.lr_scheduler.step()

        return {
            "loss": loss.item(),
        }

    def get_action(self, observation: np.ndarray) -> int:
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]

        best_action = torch.argmax(self.model(observation)).item()

        return best_action