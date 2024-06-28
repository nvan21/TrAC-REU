import torch
import numpy as np
from gymnasium import spaces
import math

from typing import Tuple

from utils import hook_fn


class PendulumEnv:
    def __init__(self, num_envs: int, device: torch.device):
        # Tensor device
        self.device = device

        # Number of actions and observations for the environment (2 for the obs because only need to store angle and angular velocity)
        self.num_obs = 2
        self.num_actions = 1

        self.num_envs = num_envs

        # Maximum values for observation and action spaces
        self.max_torque = 2.0
        self.max_speed = 8.0
        self.max_starting_speed = 1.0
        self.max_starting_pi = math.pi
        self.max_timesteps = 200

        # Create observation and action spaces
        high_obs = np.array((1.0, 1.0, self.max_speed))
        self.action_space = spaces.Box(
            low=np.array((-self.max_torque,)), high=np.array((self.max_torque,))
        )
        self.observation_space = spaces.Box(
            low=-high_obs,
            high=high_obs,
        )

        # Create buffers
        self.state = torch.empty(
            (num_envs, self.num_obs), dtype=torch.float32, device=self.device
        )
        self.reward = torch.empty(num_envs, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(num_envs, device=self.device)
        self.truncateds = torch.zeros(num_envs, device=self.device)

        # Dynamics variables
        self.g = 10.0  # m/s^2
        self.m = 1.0  # kg
        self.l = 1.0  # m
        self.dt = 0.05  # s

        # Metrics
        self.step_count = 0

    def _normalize_theta(self, theta: torch.Tensor) -> torch.Tensor:
        return ((theta + torch.pi) % (2 * torch.pi)) - torch.pi

    @torch.no_grad()
    def _get_obs(self) -> torch.Tensor:
        thetas = self.state[:, 0].unsqueeze(1)
        theta_dots = self.state[:, 1].unsqueeze(1)

        return torch.hstack((torch.cos(thetas), torch.sin(thetas), theta_dots)).to(
            self.device
        )

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        with torch.autograd.set_detect_anomaly(True):
            thetas, theta_dots = self.state[:, 0].unsqueeze(1), self.state[
                :, 1
            ].unsqueeze(1)

            self.step_count += 1
            info = {}

            g, m, l, dt = self.g, self.m, self.l, self.dt
            normalized_thetas = self._normalize_theta(thetas)

            torques = torch.clip(action, -self.max_torque, self.max_torque)
            rewards = -(
                normalized_thetas**2 + 0.1 * theta_dots**2 + 0.001 * torques**2
            ).flatten()

            # fmt: off
            new_theta_dots = theta_dots + (3 * g / (2 * l) * torch.sin(thetas) + 3.0 / (m * l*2) * torques) * dt
            # fmt: on
            new_theta_dots = torch.clip(new_theta_dots, -self.max_speed, self.max_speed)
            new_thetas = thetas + new_theta_dots * dt

            self.state = torch.hstack((new_thetas, new_theta_dots)).to(self.device)
            new_states = self._get_obs()

            if self.step_count == self.max_timesteps:
                info["final_observation"] = new_states
                self.truncateds[:] = 1.0
                new_states, _ = self.reset()
                self.step_count = 0

            return (
                new_states,
                rewards,
                self.dones,
                self.truncateds,
                info,
            )

    def reset(self) -> Tuple[torch.Tensor, dict]:
        # Sample new states from uniform distribution
        thetas = (2 * torch.pi) * torch.rand((self.num_envs, 1)) - torch.pi
        theta_dots = (2 * torch.rand((self.num_envs, 1))) - 1

        self.state = torch.hstack([thetas, theta_dots]).to(self.device)
        return self._get_obs(), {}

    def clear_grad(self):
        with torch.no_grad():
            # thetas = self.thetas.clone()
            # theta_dots = self.theta_dots.clone()
            # rewards_buf = self.rewards_buf.clone()
            # self.thetas = thetas.clone()
            # self.theta_dots = theta_dots.clone()
            # self.rewards_buf = rewards_buf.clone()
            # state = self.state.clone()
            # self.state = state.clone()
            self.state.detach_()
