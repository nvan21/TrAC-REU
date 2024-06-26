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
        num_obs = 2
        num_actions = 1

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

        # Create rollout buffers
        self.step_count = 0
        self.thetas = torch.zeros(
            (num_envs, 1), dtype=torch.float32, requires_grad=False
        ).to(device)
        self.theta_dots = torch.zeros(
            (num_envs, 1), dtype=torch.float32, requires_grad=False
        ).to(device)
        self.rewards_buf = torch.zeros(
            (num_envs, 1), dtype=torch.float32, requires_grad=False
        ).to(device)
        self.dones_buf = torch.zeros(
            (num_envs), dtype=torch.float32, requires_grad=False
        ).to(device)
        self.truncateds_buf = torch.zeros(
            (num_envs), dtype=torch.float32, requires_grad=False
        ).to(device)

        # self.thetas.register_hook(hook_fn)
        # self.theta_dots.register_hook(hook_fn)
        # self.rewards_buf.register_hook(hook_fn)

        # Initialize dynamics variables
        self.g = 10.0  # m/s^2
        self.m = 1.0  # kg
        self.l = 1.0  # m
        self.dt = 0.05  # s

    def _normalize_theta(self, theta: torch.Tensor):
        return ((theta + torch.pi) % (2 * torch.pi)) - torch.pi

    def _get_obs(self):
        x = torch.cos(self.thetas)
        y = torch.sin(self.thetas)

        obs = torch.hstack((x, y, self.theta_dots)).to(self.device)
        return obs

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        with torch.autograd.set_detect_anomaly(True):
            info = {}

            thetas, theta_dots = self.thetas, self.theta_dots
            g, m, l, dt = self.g, self.m, self.l, self.dt
            normalized_thetas = self._normalize_theta(thetas)

            torques = torch.clip(action, -self.max_torque, self.max_torque)
            self.rewards_buf = -(
                normalized_thetas**2 + 0.1 * theta_dots**2 + 0.001 * torques**2
            )
            rewards = -(normalized_thetas**2 + 0.1 * theta_dots**2 + 0.001 * torques**2)

            # fmt: off
            new_theta_dots = theta_dots + (3 * g / (2 * l) * torch.sin(thetas) + 3.0 / (m * l*2) * torques) * dt
            # fmt: on
            new_theta_dots = torch.clip(new_theta_dots, -self.max_speed, self.max_speed)
            new_thetas = thetas + new_theta_dots * dt

            self.thetas = new_thetas
            self.theta_dots = new_theta_dots

            # Add 1 timestep to the total steps
            self.step_count += 1

            # Reset the environments if the max timesteps has been reached
            if self.step_count == self.max_timesteps:
                info["final_observation"] = self._get_obs()
                reset_state, _ = self.reset()
                self.truncateds_buf = torch.ones_like(self.truncateds_buf).to(
                    self.device
                )

                return (
                    reset_state,
                    self.rewards_buf.flatten(),
                    self.dones_buf,
                    self.truncateds_buf,
                    info,
                )

            return (
                self._get_obs(),
                self.rewards_buf.flatten(),
                self.dones_buf,
                self.truncateds_buf,
                info,
            )

    def reset(self) -> Tuple[torch.Tensor, dict]:
        # Sample new states from uniform distribution
        with torch.no_grad():
            self.thetas.uniform_(-self.max_starting_pi, self.max_starting_pi)
            self.theta_dots.uniform_(-self.max_starting_speed, self.max_starting_speed)

        return self._get_obs(), {}

    def clear_grad(self):
        with torch.no_grad():
            # thetas = self.thetas.clone()
            # theta_dots = self.theta_dots.clone()
            # rewards_buf = self.rewards_buf.clone()
            # self.thetas = thetas.clone()
            # self.theta_dots = theta_dots.clone()
            # self.rewards_buf = rewards_buf.clone()
            self.thetas = self.thetas.clone()
            self.theta_dots = self.theta_dots.clone()
            self.rewards_buf = self.rewards_buf.clone()
            # self.thetas.detach()
            # self.theta_dots.detach()
            # self.rewards_buf.detach()
