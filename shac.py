from actor import StochasticActor
from critic import Critic
from hyperparameters import Config
import utils

import torch
import numpy as np
from envs import PendulumEnv
import wandb

from typing import Tuple
import pickle


class SHAC:
    def __init__(self, config: Config, envs: PendulumEnv):
        self.device = config.device
        self.config = config

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
                envs.observation_space.shape[0],
            ),
            dtype=torch.float32,
        ).to(self.device)
        self.rewards_buf = torch.zeros(
            (config.num_steps, config.num_envs),
            dtype=torch.float32,
        ).to(self.device)
        self.done_mask = torch.zeros(
            (config.num_steps, config.num_envs), dtype=torch.float32
        ).to(self.device)
        self.next_values = torch.zeros(
            (config.num_steps, config.num_envs), dtype=torch.float32
        ).to(self.device)
        self.target_values = torch.zeros(
            (config.num_steps, config.num_envs), dtype=torch.float32
        ).to(self.device)

        # Rollout tracking metrics
        self.total_timesteps = 0
        self.episode_reward = 0

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
        Takes in the returns and calculates the actor loss
        """
        return rets / -(self.config.num_envs * self.config.num_steps)

    def compute_critic_loss(self):
        """
        Computes the critic target values and then uses those to calculate the critic loss
        """
        with torch.no_grad():
            Ai = torch.zeros(
                self.config.num_envs, dtype=torch.float32, device=self.device
            )
            Bi = torch.zeros(
                self.config.num_envs, dtype=torch.float32, device=self.device
            )
            lam = torch.ones(
                self.config.num_envs, dtype=torch.float32, device=self.device
            )
            for i in reversed(range(self.config.num_steps)):
                lam = (
                    lam * self.config.gae_lambda * (1.0 - self.done_mask[i])
                    + self.done_mask[i]
                )
                Ai = (1.0 - self.done_mask[i]) * (
                    self.config.gae_lambda * self.config.gamma * Ai
                    + self.config.gamma * self.next_values[i]
                    + (1.0 - lam) / (1.0 - self.config.gae_lambda) * self.rewards_buf[i]
                )
                Bi = (
                    self.config.gamma
                    * (
                        self.next_values[i] * self.done_mask[i]
                        + Bi * (1.0 - self.done_mask[i])
                    )
                    + self.rewards_buf[i]
                )
                self.target_values[i] = (1.0 - self.config.gae_lambda) * Ai + lam * Bi

        # TODO: Implement mini batches for critic training
        total_critic_loss = 0.0
        for i in range(self.config.critic_iterations):
            predicted_values = self.critic(
                self.states_buf.view(-1, self.states_buf.shape[-1])
            ).squeeze(-1)
            target_values = self.target_values.view(-1)
            total_critic_loss += ((predicted_values - target_values) ** 2).mean()

        return total_critic_loss / self.config.critic_iterations

    def train(self):
        states, _ = self.envs.reset()

        critic_loss_hist = []
        actor_loss_hist = []
        for epoch in range(self.config.max_epochs):
            rets, states = self.rollout(states)

            actor_loss = self.compute_actor_loss(rets)
            critic_loss = self.compute_critic_loss()

            # Get the updated learning rates
            actor_lr = self.actor_lr_schedule[epoch]
            critic_lr = self.critic_lr_schedule[epoch]

            # Backpropogate the network losses
            self.actor.backward(loss=actor_loss, learning_rate=actor_lr)
            self.critic.backward(loss=critic_loss, learning_rate=critic_lr)

            print(f"Epoch: ({epoch})")
            print(f"Timestep: {self.total_timesteps}")
            print(f"actor loss: {actor_loss}")
            print(f"critic loss: {critic_loss}")
            print(f"episode reward: {self.episode_reward / self.config.num_envs}")
            print("")

            if self.config.do_wandb_logging:
                wandb.log(
                    {
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                    }
                )

            actor_loss_hist.append(actor_loss.detach().cpu())
            critic_loss_hist.append(critic_loss.detach().cpu())

            self.soft_network_update()
            self.save()

        print(np.array(actor_loss_hist[-20:]).mean())
        print(np.array(critic_loss_hist[-20:]).mean())

    def rollout(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs num_steps rollouts. It takes in the intial states and then returns the GAE returns
        and the final states of each environment.
        """
        with torch.autograd.set_detect_anomaly(True):
            running_rewards = torch.zeros(
                (self.config.num_steps + 1, self.config.num_envs), dtype=torch.float32
            ).to(self.device)
            gamma = torch.ones(self.config.num_envs, dtype=torch.float32).to(
                self.device
            )
            next_values = torch.zeros(
                (self.config.num_steps + 1, self.config.num_envs), dtype=torch.float32
            ).to(self.device)

            actor_loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)

            self.envs.clear_grad()
            for step in range(self.config.num_steps):
                with torch.no_grad():
                    # Add the states to the replay buffer (used for critic training)
                    self.states_buf[step] = states

                # Get the actions from the actor and step through the environment
                actions = self.actor(states)
                states, rewards, dones, truncateds, info = self.envs.step(actions)

                # Get the IDs of all the environments that terminated/truncated this step
                done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
                truncated_ids = torch.nonzero(truncateds, as_tuple=False).squeeze(-1)

                # Get the estimated critic value of the next state, and set the terminated states to 0
                next_values[step + 1] = self.target_critic(states).squeeze(-1)
                next_values[step + 1, done_ids] = 0.0

                # Change the next value of truncated states to use their final observation rather than the reset observation
                if ("final_observation") in info and len(truncated_ids) != 0:
                    # Mask out the None values to get the truncated states
                    truncated_states = info["final_observation"]
                    truncated_states = [
                        arr for arr in truncated_states if arr is not None
                    ]

                    # Turn the array of arrays of states into a single 2D tensor
                    truncated_states = np.vstack(truncated_states)
                    truncated_states = utils.np_to_tensor(
                        device=self.device, arrays=[truncated_states]
                    )

                    # Updated the next values to accurately reflect the final observation of the truncated state
                    next_values[step + 1, truncated_ids] = self.target_critic(
                        truncated_states
                    ).squeeze(-1)

                    if self.config.do_wandb_logging:
                        # Log and reset episode rewards
                        wandb.log(
                            {
                                "episode_rewards": self.episode_reward
                                / self.config.num_envs
                            }
                        )
                        self.episode_reward = 0

                # Update the running rewards
                running_rewards[step + 1, :] = (
                    running_rewards[step, :] + self.config.gamma * rewards
                )

                # Get the estimated value of all states for the last timestep in the trajectory
                if step < self.config.num_steps - 1:
                    actor_loss = actor_loss + (
                        (
                            (
                                running_rewards[step + 1, done_ids]
                                + self.config.gamma
                                * gamma[done_ids]
                                * next_values[step + 1, done_ids]
                            )
                        ).sum()
                    )
                else:
                    actor_loss = (
                        actor_loss
                        + (
                            running_rewards[step + 1, :]
                            + self.config.gamma * gamma * next_values[step + 1, :]
                        ).sum()
                    )

                # Calculate gamma for the next step
                gamma = gamma * self.config.gamma

                # Reset gamma and returns for done environments
                gamma[done_ids] = 1.0
                running_rewards[step + 1, done_ids] = 0.0

                with torch.no_grad():
                    # Update data for critic training
                    self.rewards_buf[step] = rewards.clone()

                    if step < self.config.num_steps - 1:
                        # Update the done mask for the terminated environments
                        self.done_mask[step] = dones.clone().to(torch.float32)
                    else:
                        self.done_mask[step, :] = 1.0

                self.total_timesteps += self.config.num_envs
                self.episode_reward += rewards.sum().item()

            return actor_loss, states

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

    def soft_network_update(self):
        params = self.critic.parameters()
        target_params = self.target_critic.parameters()

        for param, target_param in zip(params, target_params):
            target_param.data.copy_(
                target_param.data * self.config.tau + (1 - self.config.tau) * param.data
            )

    def save(self, filename=None):
        if filename is None:
            filename = "best_policy.pt"

        torch.save([self.actor, self.critic, self.target_critic], filename)

    def load(self, filename=None):
        if filename is None:
            filename = "best_policy.pt"

        checkpoint = torch.load(filename)
        self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.target_critic = checkpoint[2].to(self.device)
