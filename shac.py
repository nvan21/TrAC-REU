from actor import StochasticActor
from critic import Critic
from hyperparameters import PendulumParams
import utils

import torch
import numpy as np
from envs import PendulumEnv
import wandb

from typing import Tuple
from datetime import datetime
import os
from collections import deque


class SHAC:
    def __init__(self, params: PendulumParams, envs: PendulumEnv):
        self.device = params.device
        self.params = params
        self.envs = envs

        # Create learning rate schedules
        self.actor_lr_schedule = self.learning_rate_scheduler(
            lr=self.params.actor_learning_rate,
            decay=self.params.actor_learning_rate_schedule,
        )
        self.critic_lr_schedule = self.learning_rate_scheduler(
            lr=self.params.critic_learning_rate,
            decay=self.params.critic_learning_rate_schedule,
        )

        # Replay buffers
        self.states_buf = torch.zeros(
            (
                params.num_steps,
                params.num_envs,
                envs.observation_space.shape[0],
            ),
            dtype=torch.float32,
        ).to(self.device)
        self.rewards_buf = torch.zeros(
            (params.num_steps, params.num_envs),
            dtype=torch.float32,
        ).to(self.device)
        self.done_mask = torch.zeros(
            (params.num_steps, params.num_envs), dtype=torch.float32
        ).to(self.device)
        self.next_values = torch.zeros(
            (params.num_steps, params.num_envs), dtype=torch.float32
        ).to(self.device)
        self.target_values = torch.zeros(
            (params.num_steps, params.num_envs), dtype=torch.float32
        ).to(self.device)

        # Rollout tracking metrics
        self.total_timesteps = 0
        self.episode_reward = 0
        self.best_actor_loss = torch.inf
        self.actor_loss_hist = deque(maxlen=10)
        self.critic_loss_hist = deque(maxlen=10)

        # Date for the filename
        self.date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def create_models(
        self,
        act_dim: int,
        obs_dim: int,
    ) -> None:
        self.actor = StochasticActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            units=self.params.actor_units,
            activation_fn=self.params.actor_activation,
            device=self.device,
        )

        self.critic = Critic(
            obs_dim=obs_dim,
            units=self.params.critic_units,
            activation_fn=self.params.critic_activation,
            device=self.device,
        )

        self.target_critic = Critic(
            obs_dim=obs_dim,
            units=self.params.critic_units,
            activation_fn=self.params.critic_activation,
            device=self.device,
        )

        self.hard_network_update()

    def compute_actor_loss(self, rets: torch.Tensor):
        """
        Takes in the returns and calculates the actor loss
        """
        return rets / -(self.params.num_envs * self.params.num_steps)

    def compute_critic_loss(self, learning_rate: float):
        """
        Computes the critic target values and then uses those to calculate the critic loss
        """
        with torch.no_grad():
            Ai = torch.zeros(
                self.params.num_envs, dtype=torch.float32, device=self.device
            )
            Bi = torch.zeros(
                self.params.num_envs, dtype=torch.float32, device=self.device
            )
            lam = torch.ones(
                self.params.num_envs, dtype=torch.float32, device=self.device
            )
            for i in reversed(range(self.params.num_steps)):
                lam = (
                    lam * self.params.gae_lambda * (1.0 - self.done_mask[i])
                    + self.done_mask[i]
                )
                Ai = (1.0 - self.done_mask[i]) * (
                    self.params.gae_lambda * self.params.gamma * Ai
                    + self.params.gamma * self.next_values[i]
                    + (1.0 - lam) / (1.0 - self.params.gae_lambda) * self.rewards_buf[i]
                )
                Bi = (
                    self.params.gamma
                    * (
                        self.next_values[i] * self.done_mask[i]
                        + Bi * (1.0 - self.done_mask[i])
                    )
                    + self.rewards_buf[i]
                )
                self.target_values[i] = (1.0 - self.params.gae_lambda) * Ai + lam * Bi

        total_critic_loss = 0.0
        for i in range(self.params.critic_iterations):
            # Create a tensor with randomized indexes
            idxs = torch.randperm(self.params.num_steps).view(
                self.params.critic_minibatches, -1
            )
            states = self.states_buf.view(-1, self.states_buf.shape[-1])
            target_values = self.target_values.view(-1)

            for j in range(self.params.critic_minibatches):
                idx = idxs[j]
                mb_states = states[idx]
                mb_predicted_values = self.critic(mb_states).squeeze(-1)
                mb_target_values = target_values[idx].view(-1)

                critic_loss = ((mb_predicted_values - mb_target_values) ** 2).mean()

                self.critic.backward(loss=critic_loss, learning_rate=learning_rate)

                total_critic_loss += critic_loss.item()

        return total_critic_loss / self.params.critic_iterations

    def train(self):
        states, _ = self.envs.reset()

        for epoch in range(self.params.max_epochs):
            rets, states = self.rollout(states)

            # Get the updated learning rates
            actor_lr = self.actor_lr_schedule[epoch]
            critic_lr = self.critic_lr_schedule[epoch]

            # Calculate and backpropogate the actor loss
            actor_loss = self.compute_actor_loss(rets)
            self.actor.backward(loss=actor_loss, learning_rate=actor_lr)

            # Calculate and backpropogate the critic loss (it's backpropogated in the compute method)
            critic_loss = self.compute_critic_loss(learning_rate=critic_lr)

            # Store the losses for the average loss calculation
            self.actor_loss_hist.append(actor_loss)
            self.critic_loss_hist.append(critic_loss)

            # Save the model if it has the best loss
            if actor_loss < self.best_actor_loss:
                self.save()

            # Debugging/logging
            avg_actor_loss = sum(self.actor_loss_hist) / len(self.actor_loss_hist)
            avg_critic_loss = sum(self.critic_loss_hist) / len(self.critic_loss_hist)
            print(f"Epoch: ({epoch})")
            print(f"Timestep: {self.total_timesteps}")
            print(f"Average actor loss: {avg_actor_loss}")
            print(f"critic loss: {avg_critic_loss} \n")

            if self.params.do_wandb_logging:
                wandb.log(
                    {
                        "actor_loss": avg_actor_loss,
                        "critic_loss": avg_critic_loss,
                    }
                )

            # Update the target network
            self.soft_network_update()

    def evaluate_policy(self):
        state, _ = self.envs.reset()
        done = False
        truncated = False

        while not done and not truncated:
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, device=self.device)

            action = self.actor(state).detach().numpy()
            state, reward, done, truncated, _ = self.envs.step(action)

            self.episode_reward += reward

        print(f"final evaluation reward: {self.episode_reward}")

    def rollout(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs num_steps rollouts. It takes in the intial states and then returns the GAE returns
        and the final states of each environment.
        """
        with torch.autograd.set_detect_anomaly(True):
            running_rewards = torch.zeros(
                (self.params.num_steps + 1, self.params.num_envs), dtype=torch.float32
            ).to(self.device)
            gamma = torch.ones(self.params.num_envs, dtype=torch.float32).to(
                self.device
            )
            next_values = torch.zeros(
                (self.params.num_steps + 1, self.params.num_envs), dtype=torch.float32
            ).to(self.device)

            actor_loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)

            self.envs.clear_grad()
            for step in range(self.params.num_steps):
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
                    # Get the states before the environments were reset
                    truncated_states = info["final_observation"]

                    # Updated the next values to accurately reflect the final observation of the truncated state
                    next_values[step + 1, truncated_ids] = self.target_critic(
                        truncated_states
                    ).squeeze(-1)

                    if self.params.do_wandb_logging:
                        # Log and reset episode rewards
                        wandb.log(
                            {
                                "episode_rewards": self.episode_reward
                                / self.params.num_envs
                            }
                        )
                        print(
                            f"Episode has ended! Average episode reward: {self.episode_reward / self.params.num_envs}\n"
                        )
                    self.episode_reward = 0

                # Update the running rewards
                running_rewards[step + 1, :] = (
                    running_rewards[step, :] + self.params.gamma * rewards
                )

                # Get the estimated value of all states for the last timestep in the trajectory
                if step < self.params.num_steps - 1:
                    actor_loss = actor_loss + (
                        (
                            (
                                running_rewards[step + 1, done_ids]
                                + self.params.gamma
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
                            + self.params.gamma * gamma * next_values[step + 1, :]
                        ).sum()
                    )

                # Calculate gamma for the next step
                gamma = gamma * self.params.gamma

                # Reset gamma and returns for done environments
                gamma[done_ids] = 1.0
                running_rewards[step + 1, done_ids] = 0.0

                with torch.no_grad():
                    # Update data for critic training
                    self.rewards_buf[step] = rewards.clone()

                    if step < self.params.num_steps - 1:
                        # Update the done mask for the terminated environments
                        self.done_mask[step] = dones.clone().to(torch.float32)
                    else:
                        self.done_mask[step, :] = 1.0

                    self.episode_reward += rewards.sum().item()

                self.total_timesteps += self.params.num_envs

            return actor_loss, states

    def learning_rate_scheduler(self, lr: float, decay: str) -> list:
        if decay == "constant":
            schedule = [lr for _ in range(self.params.max_epochs)]
        elif decay == "linear":
            schedule = [
                lr * (self.params.max_epochs - i) / self.params.max_epochs
                for i in range(self.params.max_epochs)
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
                target_param.data * self.params.tau + (1 - self.params.tau) * param.data
            )

    def save(self, filename=None):
        if filename is None:
            filename = f"weights/{self.date}"

            if not os.path.exists(filename):
                os.makedirs(filename)

        torch.save(
            [self.actor, self.critic, self.target_critic], f"{filename}/best_policy.pt"
        )

    def load(self, filename: str) -> None:
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor = checkpoint[0].to(self.device)
        self.critic = checkpoint[1].to(self.device)
        self.target_critic = checkpoint[2].to(self.device)
