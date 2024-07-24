# import os
# import gymnasium as gym
# import torch
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# import pandas as pd

# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.monitor import Monitor


# # Set Seed Function
# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# # Custom Environment Wrapper
# class SeededEnvWrapper(gym.Wrapper):
#     def __init__(self, env, seed=None):
#         super(SeededEnvWrapper, self).__init__(env)
#         self.seed_value = seed

#     def reset(self, **kwargs):
#         if self.seed_value is not None:
#             self.env.action_space.seed(self.seed_value)
#             self.env.observation_space.seed(self.seed_value)
#             set_seed(self.seed_value)
#         return self.env.reset(**kwargs)

#     def step(self, action):
#         return self.env.step(action)


# # Create Seeded Environment Function
# def create_seeded_env(env_id, seed, log_dir, rank=0):
#     env = gym.make(env_id)
#     env = SeededEnvWrapper(env, seed=seed)
#     env = Monitor(
#         env, filename=os.path.join(log_dir, f"monitor_{rank}.csv")
#     )  # Wrap the environment with Monitor
#     return env


# # Function to make environments
# def make_env(env_id, seed, rank, log_dir):
#     def _init():
#         env = create_seeded_env(env_id, seed + rank, log_dir, rank)
#         return env

#     set_seed(seed)
#     return _init


# # Plotting Function
# def plot_learning_curve(log_dir, title="Learning Curve"):
#     monitor_files = [
#         os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("monitor")
#     ]
#     df_list = [pd.read_csv(f, skiprows=1) for f in monitor_files]

#     # Ensure all DataFrames have the same length by truncating to the shortest one
#     min_length = min(len(df) for df in df_list)
#     df_list = [df.iloc[:min_length] for df in df_list]

#     # Concatenate cumulative timesteps
#     cumulative_timesteps = np.concatenate([np.cumsum(df["l"].values) for df in df_list])

#     # Combine the DataFrames by averaging the 'r' values for each episode
#     rewards = np.vstack([df["r"].values for df in df_list]).flatten()
#     mean_rewards = []
#     std_rewards = []
#     unique_timesteps = np.unique(cumulative_timesteps)

#     for t in unique_timesteps:
#         mask = cumulative_timesteps == t
#         mean_rewards.append(rewards[mask].mean())
#         std_rewards.append(rewards[mask].std())

#     mean_rewards = np.array(mean_rewards)
#     std_rewards = np.array(std_rewards)
#     unique_timesteps *= 4

#     # Plot the learning curve with the shaded area representing the range
#     plt.figure(figsize=(12, 8))
#     plt.plot(
#         unique_timesteps, mean_rewards, label="Episode Reward (Mean)", color="blue"
#     )
#     plt.fill_between(
#         unique_timesteps,
#         mean_rewards - std_rewards,
#         mean_rewards + std_rewards,
#         color="blue",
#         alpha=0.3,
#     )
#     plt.xlabel("Timesteps", fontsize=14)
#     plt.ylabel("Return", fontsize=14)
#     plt.title(title, fontsize=16)
#     plt.legend(loc="upper left", fontsize=12)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#     plt.savefig("test.png", format="png")


# # Main Training Script
# if __name__ == "__main__":
#     # env_id = "Pendulum-v1"
#     # seed = 42
#     # num_envs = 4  # Number of parallel environments
#     log_dir = "./logs/"
#     os.makedirs(log_dir, exist_ok=True)

#     # # Create vectorized environment
#     # envs = [make_env(env_id, seed, i, log_dir) for i in range(num_envs)]
#     # vec_env = DummyVecEnv(envs)  # Use DummyVecEnv for simplicity

#     # # Create and train the PPO model
#     # model = PPO("MlpPolicy", vec_env, verbose=1)
#     # model.learn(total_timesteps=500000)

#     # # Save the model
#     # model.save("ppo_pendulum_seeded_vec")

#     # Evaluate the model
#     # obs = vec_env.reset()
#     # for _ in range(1000):
#     #     action, _states = model.predict(obs)
#     #     obs, rewards, dones, infos = vec_env.step(action)
#     #     if np.any(dones):
#     #         obs = vec_env.reset()

#     # Plot the learning curve
#     plot_learning_curve(log_dir)
import os
import gymnasium as gym
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from hyperparameters import PPOPendulumParams

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


# Set Seed Function
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Custom Environment Wrapper
class SeededEnvWrapper(gym.Wrapper):
    def __init__(self, env, seed=None):
        super(SeededEnvWrapper, self).__init__(env)
        self.seed_value = seed

    def reset(self, **kwargs):
        if self.seed_value is not None:
            self.env.action_space.seed(self.seed_value)
            self.env.observation_space.seed(self.seed_value)
            set_seed(self.seed_value)
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)


# Create Seeded Environment Function
def create_seeded_env(env_id, seed, log_dir, rank=0):
    env = gym.make(env_id)
    env = SeededEnvWrapper(env, seed=seed)
    env = Monitor(
        env,
        filename=os.path.join(log_dir, f"monitor_{seed}_{rank}.csv"),
        allow_early_resets=True,
    )
    return env


# Function to make environments
def make_env(env_id, seed, rank, log_dir):
    def _init():
        env = create_seeded_env(env_id, seed + rank, log_dir, rank)
        return env

    set_seed(seed)
    return _init


# Main Training Script
if __name__ == "__main__":
    env_id = "Pendulum-v1"
    params = PPOPendulumParams()
    seeds = params.seeds  # List of seeds
    num_envs = params.num_envs  # Number of parallel environments

    for seed in seeds:
        log_dir = f"./experiments/training_curves/PPO/seed_{seed}"
        os.makedirs(log_dir, exist_ok=True)
        # Create vectorized environment
        envs = [make_env(env_id, seed, i, log_dir) for i in range(num_envs)]
        vec_env = DummyVecEnv(envs)  # Use DummyVecEnv for simplicity

        # Create and train the PPO model
        model = PPO("MlpPolicy", vec_env, verbose=1)
        model.learn(total_timesteps=params.max_timesteps)

        # Save the model
        model.save(f"{log_dir}/best_policy")
