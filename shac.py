from actor import StochasticActor
from critic import Critic
from hyperparameters import Config

import torch
import numpy as np


class SHAC:
    def __init__(self, config: Config):
        self.device = config.device
        self.config = config

    def create_models(
        self,
        act_dim: int,
        obs_dim: int,
    ):
        self.actor = StochasticActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            units=self.config.actor_units,
            activation_fn=self.config.actor_activation,
            device=self.device,
        )

        self.critic = Critic(
            obs_dim=obs_dim,
            units=self.config.critic_units,
            activation_fn=self.config.critic_activation,
            device=self.device,
        )

    def get_action(self, obs) -> torch.Tensor:
        if type(obs) is np.ndarray:
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        return self.actor(obs)

    def get_value(self, obs) -> torch.Tensor:
        if type(obs) is np.ndarray:
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        return self.critic(obs)
