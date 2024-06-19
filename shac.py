from actor import StochasticActor
from critic import Critic
from hyperparameters import Config
import utils

import torch
import numpy as np
from gymnasium.vector import VectorEnv

from typing import Tuple


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
        self.done_mask = torch.zeros(
            (config.num_steps, config.num_envs), dtype=torch.float32
        ).to(self.device)

        # Rollout tracking metrics
        self.episode_length = 0

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

    def compute_actor_loss(self, rets: torch.Tensor):
        """
        Takes in the GAE returns and calculates the actor loss
        """
        rets *= self.done_mask

        return rets.sum() / (-self.config.num_envs * self.config.num_steps)

    def compute_critic_loss(self, rets: torch.Tensor):
        pass

    def train(self):
        states, _ = self.envs.reset()
        states = utils.np_to_tensor(device=self.device, arrays=[states])

        for epoch in range(1):  # self.config.max_epochs):
            rets, states = self.rollout(states)
            actor_loss = self.compute_actor_loss(rets)

            # TODO: Implement critic loss
            critic_loss = self.compute_critic_loss()

            # Get the updated learning rates
            actor_lr = self.actor_lr_schedule[epoch]
            critic_lr = self.critic_lr_schedule[epoch]

            # Backpropogate the network losses
            self.actor.backward(loss=actor_loss, learning_rate=actor_lr)
            self.critic.backward(loss=critic_loss, learning_rate=critic_lr)

            print(actor_loss)
            print(critic_loss)

    def rollout(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs num_steps rollouts. It takes in the intial states and then returns the GAE returns
        and the final states of each environment.
        """
        gamma = torch.ones(self.config.num_envs, dtype=torch.float32).to(self.device)

        terminal_values = torch.zeros(
            (self.config.num_steps + 1, self.config.num_envs), dtype=torch.float32
        ).to(self.device)
        rets = torch.zeros(
            (self.config.num_steps + 1, self.config.num_envs), dtype=torch.float32
        ).to(self.device)
        done_mask = torch.zeros(
            (self.config.num_steps, self.config.num_envs), dtype=torch.float32
        ).to(self.device)

        for step in range(self.config.num_steps):
            actions = self.actor(states).detach().cpu().numpy()
            states, rewards, dones, truncateds, info = self.envs.step(actions)

            states, rewards, dones, truncateds = utils.np_to_tensor(
                device=self.device, arrays=[states, rewards, dones, truncateds]
            )

            # Add the states and rewards to the replay buffer (used for critic training)
            self.states_buf[step] = states
            self.rewards_buf[step] = rewards

            # Get the IDs of all the environments that terminated/truncated this step
            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            truncated_ids = torch.nonzero(truncateds, as_tuple=False).squeeze(-1)

            # Change the next value of truncated states to use their final observation rather than the reset observation
            if ("final_observation") in info and len(truncated_ids) != 0:
                # Mask out the None values to get the truncated states
                truncated_states = info["final_observation"]
                truncated_states[truncated_states == None] = 0
                truncated_states = utils.np_to_tensor(
                    device=self.device, arrays=[truncated_states]
                )
                truncated_states = torch.nonzero(truncated_states, as_tuple=False)

                # Updated the next values to accurately reflect the final observation of the truncated state
                terminal_values[step + 1, truncated_ids] = self.target_critic(
                    truncated_states
                ).squeeze(-1)

            # Update the done mask for the terminated environments
            self.done_mask[step, done_ids] = 1.0

            # Get the estimated value of all states for the last timestep in the trajectory
            if step == self.config.num_steps - 1:
                terminal_values[step + 1, :] = self.target_critic(states).squeeze(-1)

            # Update the returns - the terminal values will be 0 unless it's the last timestep in the trajectory
            rets[step + 1, :] = (
                rets[step, :]
                + rewards * gamma
                + self.config.gamma * gamma * terminal_values[step + 1, :]
            )
            # Calculate gamma for the next step
            gamma = gamma * self.config.gamma

            # Reset gamma and returns for done environments
            gamma[done_ids] = 1.0
            rets[step + 1, done_ids] = 0.0

        # Get rid of the first row since it's no longer needed
        rets = rets[1:, :]
        self.done_mask[-1, :] = 1.0

        return rets, states

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
