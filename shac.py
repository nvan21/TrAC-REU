from actor import StochasticActor
from critic import Critic
from hyperparameters import Config

import torch
import numpy as np
from gymnasium.vector import VectorEnv


class SHAC:
    def __init__(self, config: Config, envs: VectorEnv):
        self.device = config.device
        self.config = config
        self.envs = envs

        # Create learning rate schedules
        self.actor_lr_schedule = self.learning_rate_scheduler(
            lr=self.config.actor_learning_rate,
            decay=self.config.actor_learning_rate_schedule,
        )
        self.critic_lr_schedule = self.learning_rate_scheduler(
            lr=self.config.critic_learning_rate,
            decay=self.config.critic_learning_rate_schedule,
        )

        # Replay buffers
        self.states_buf = torch.zeros(
            (
                config.num_steps,
                config.num_envs,
                envs.unwrapped.observation_space.shape[1],
            ),
            dtype=torch.float32,
        ).to(self.device)
        self.actions_buf = torch.zeros(
            (
                config.num_steps,
                config.num_envs,
                envs.unwrapped.action_space.shape[1],
            ),
            dtype=torch.float32,
        ).to(self.device)
        self.rewards_buf = torch.zeros(
            (config.num_steps, config.num_envs), dtype=torch.float32
        ).to(self.device)

    def create_models(
        self,
        act_dim: int,
        obs_dim: int,
    ) -> None:
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

        self.target_critic = Critic(
            obs_dim=obs_dim,
            units=self.config.critic_units,
            activation_fn=self.config.critic_activation,
            device=self.device,
        )

        self.hard_network_update()

    def get_action(self, obs) -> torch.Tensor:
        if type(obs) is np.ndarray:
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        return self.actor(obs)

    def get_critic_value(self, obs) -> torch.Tensor:
        if type(obs) is np.ndarray:
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        return self.critic(obs)

    def get_target_critic_value(self, obs) -> torch.Tensor:
        if type(obs) is np.ndarray:
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)

        return self.target_critic(obs)

    def train(self):
        states, _ = self.envs.reset()
        self.rollout(states)

    def rollout(self, states: np.ndarray):
        gamma = torch.ones(self.config.num_envs, dtype=torch.float32).to(self.device)
        next_values = torch.zeros(
            (self.config.num_steps + 1, self.config.num_envs), dtype=torch.float32
        ).to(self.device)
        running_rewards = torch.zeros(
            (self.config.num_steps + 1, self.config.num_envs), dtype=torch.float32
        ).to(self.device)

        actor_loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)

        for step in range(self.config.num_steps):
            actions = self.get_action(states)
            states, rewards, dones, truncateds, info = self.envs.step(
                actions.detach().cpu().numpy()
            )

            # Add the states and rewards to the replay buffer (used for critic training)
            self.states_buf[step] = torch.tensor(states, dtype=torch.float32).to(
                self.device
            )
            self.rewards_buf[step] = torch.tensor(rewards, dtype=torch.float32).to(
                self.device
            )

            # Get the IDs of all the environments that terminated/truncated this step
            done_ids = np.flatnonzero(dones)
            truncated_ids = np.flatnonzero(truncateds)

            # Calculate gamma for the next step and reset gamma for the terminated environments
            gamma = gamma * self.config.gamma

            # Naively get the critic values for all states
            next_values[step + 1, :] = self.get_target_critic_value(states).squeeze(-1)

            # Change the next value of terminal states to 0
            next_values[step + 1, done_ids] = 0.0

            print(truncated_ids)
            # Change the next value of truncated states to use their final observation rather than the reset observation
            if ("final_observation") in info and len(truncated_ids) != 0:
                # Mask out the None values to get the final states
                final_states = info["final_observation"]
                final_states[final_states == None] = 0
                final_states = np.nonzero(final_states)

                next_values[step + 1, truncated_ids] = self.get_target_critic_value(
                    final_states
                )

            # Reset gamma and running reward for done environments
            gamma[done_ids] = 1.0
            running_rewards[step + 1, done_ids] = 0.0

    def learning_rate_scheduler(self, lr: float, decay: str) -> list:
        if decay == "constant":
            schedule = [lr for _ in range(self.config.max_epochs)]
        elif decay == "linear":
            schedule = [
                lr * (self.config.max_epochs - i) / self.config.max_epochs
                for i in range(self.config.max_epochs)
            ]

        return schedule

    def hard_network_update(self):
        params = self.critic.parameters()
        target_params = self.target_critic.parameters()

        for param, target_param in zip(params, target_params):
            param.data.copy_(target_param.data)
