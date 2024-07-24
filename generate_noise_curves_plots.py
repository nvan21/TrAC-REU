import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import pickle
import numpy as np
import os
import pandas as pd


def extract_pickle_data(log_dir):
    seeds = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]
    rewards = []
    for seed in seeds:
        with open(f"{log_dir}/{seed}/data.pkl", "rb") as f:
            data = pickle.load(f)
            rewards.append(data["rewards"])
            timesteps = data["noises"]

    rewards = np.array(rewards)
    timesteps = np.array(timesteps)

    mean_rewards = rewards.mean(axis=0)
    std_rewards = rewards.std(axis=0)

    return mean_rewards, std_rewards, timesteps


def extract_rewards_from_monitors(log_dir):
    all_rewards = []
    seeds = os.listdir(log_dir)

    for seed in seeds:
        seed_rewards = []
        seed_dir = os.path.join(log_dir, seed)
        monitor_files = [
            os.path.join(seed_dir, f)
            for f in os.listdir(seed_dir)
            if f.startswith("monitor")
        ]
        num_envs = len(monitor_files)
        for monitor_file in monitor_files:
            df = pd.read_csv(monitor_file, skiprows=1)
            seed_rewards.append(df["r"].values)
            timesteps = df["l"].values

        # Convert to numpy array for easy manipulation
        seed_rewards = np.array(seed_rewards)

        # Compute the average rewards at each timestep
        avg_rewards = seed_rewards.mean(axis=0)
        all_rewards.append(avg_rewards)

    all_rewards = np.array(all_rewards)
    mean_rewards = all_rewards.mean(axis=0)
    std_rewards = all_rewards.std(axis=0)

    timesteps = np.array(timesteps)
    timesteps = np.cumsum(timesteps) * num_envs

    return mean_rewards, std_rewards, timesteps


def plot_means_and_stds(means_list, stds_list, timesteps_list, series_list):
    for idx, (means, stds) in enumerate(zip(means_list, stds_list)):
        ts = timesteps_from hyperparameters import SHACPendulumParams
from shac import SHAC
from envs import PendulumEnv

import numpy as np
import torch
import gymnasium as gym
import os
import pickle


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    # Get config
    params = SHACPendulumParams()

    # Initialize environments and SHAC instance
    envs = PendulumEnv(num_envs=params.num_envs, device=params.device)
    eval_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    eval_env.action_space.seed(0)
    eval_env.observation_space.seed(0)

    obs_dim = envs.observation_space.shape[0]
    act_dim = envs.action_space.shape[0]
    for seed in params.seeds:
        run_dir = f"experiments/noise_curves/SHAC/seed_{seed}"
        os.makedirs(run_dir, exist_ok=True)
        shac = SHAC(params=params, envs=envs, eval_env=eval_env, run_dir=run_dir)
        shac.set_seed(seed=seed)
        shac.create_models(act_dim=act_dim, obs_dim=obs_dim)
        shac.load(f"experiments/training_curves/SHAC/seed_{seed}/best_policy.pt")
        shac.actor.eval()

        set_seed(seed=seed)
        rewards = []
        std = []

        with torch.no_grad():
            for noise in params.noises:
                state, _ = eval_env.reset(seed=seed)
                done, truncated = False, False
                total_reward = 0

                while not done and not truncated:
                    state = state + np.random.normal(loc=0.0, scale=noise, size=obs_dim)
                    if isinstance(state, np.ndarray):
                        state = torch.tensor(
                            state, dtype=torch.float32, device=params.device
                        )

                    action = shac.actor(state).cpu().detach().numpy()
                    state, reward, done, truncated, _ = eval_env.step(action)

                    total_reward += reward

                print(f"Final reward for standard deviation of {noise}: {total_reward}")
                rewards.append(total_reward)
                std.append(noise)

        with open(f"{run_dir}/data.pkl", "wb") as f:
            data = {"rewards": rewards, "noises": std}
            pickle.dump(data, f)
list[idx]
        label = series_list[idx]
        plt.plot(ts, means, label=label, linewidth=1.5)
        plt.fill_between(ts, means - stds, means + stds, alpha=0.3)

    plt.title("Pendulum Robustness Comparison", fontsize=20, fontweight="bold")
    plt.xlabel(f"Noise standard deviation", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # ax = plt.gca()
    # ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))

    plt.tight_layout()
    plt.show()
    plt.savefig("experiments/noise_curves/Robustness Comparison.png", format="png")
    plt.savefig("experiments/noise_curves/Robustness Comparison.svg", format="svg")


model = "SHAC"
shac_run_dir = "experiments/noise_curves/SHAC"
ppo_run_dir = "experiments/noise_curves/PPO"
sac_run_dir = "experiments/noise_curves/SAC"

shac_mean_rewards, shac_std_rewards, shac_timesteps = extract_pickle_data(shac_run_dir)
ppo_mean_rewards, ppo_std_rewards, ppo_timesteps = extract_pickle_data(
    log_dir=ppo_run_dir
)
sac_mean_rewards, sac_std_rewards, sac_timesteps = extract_pickle_data(
    log_dir=sac_run_dir
)

plot_means_and_stds(
    means_list=[shac_mean_rewards, ppo_mean_rewards, sac_mean_rewards],
    stds_list=[shac_std_rewards, ppo_std_rewards, sac_std_rewards],
    timesteps_list=[shac_timesteps, ppo_timesteps, sac_timesteps],
    series_list=["SHAC", "PPO", "SAC"],
)
