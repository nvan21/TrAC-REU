import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib import font_manager
import pickle
import numpy as np
import os
import pandas as pd

title_font_path = "/Users/nvan/Library/Fonts/segoeuithibd.ttf"
axes_font_path = "/Users/nvan/Library/Fonts/segoeuithis.ttf"
font_manager.fontManager.addfont(title_font_path)
font_manager.fontManager.addfont(axes_font_path)

axes_font = font_manager.FontProperties(fname=axes_font_path).get_name()
title_font = font_manager.FontProperties(fname=title_font_path)

plt.rcParams["font.family"] = axes_font
plt.rcParams["font.size"] = 18
plt.rcParams["axes.titlesize"] = 32


def extract_pickle_data(log_dir):
    seeds = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]
    rewards = []
    for seed in seeds:
        with open(f"{log_dir}/{seed}/data.pkl", "rb") as f:
            data = pickle.load(f)
            rewards.append(data["rewards"])
            timesteps = data["timesteps"]

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


def plot_means_and_stds(
    means_list, stds_list, timesteps_list, series_list, colors_list
):

    plt.figure(figsize=(12, 8))
    baseline = -150
    for idx, (means, stds) in enumerate(zip(means_list, stds_list)):
        ts = timesteps_list[idx] / 1e5
        label = series_list[idx]
        plt.plot(
            ts,
            means,
            label=label,
            linewidth=1.5,
            color=colors_list[idx],
            zorder=idx - 1,
        )
        plt.fill_between(
            ts,
            means - stds,
            means + stds,
            alpha=0.2,
            color=colors_list[idx],
            zorder=idx - 1,
        )

    baseline_ts = plt.gca().get_xlim()
    plt.plot(
        [baseline_ts[0], baseline_ts[-1]],
        [baseline, baseline],
        label="Expert level performance",
        linewidth=1.5,
        linestyle="--",
        color="black",
    )
    plt.title("Pendulum Learning Comparison", font=title_font, fontsize=40)
    plt.xlabel(f"Simulation steps (x10\u2075)")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.xlim(baseline_ts)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # plt.tight_layout()
    plt.savefig(
        "experiments/training_curves/Learning Curve Comparison.png", format="png"
    )
    plt.savefig(
        "experiments/training_curves/Learning Curve Comparison.svg", format="svg"
    )


model = "SHAC"
shac_run_dir = "experiments/training_curves/SHAC"
ppo_run_dir = "experiments/training_curves/PPO"
sac_run_dir = "experiments/training_curves/SAC"

shac_mean_rewards, shac_std_rewards, shac_timesteps = extract_pickle_data(shac_run_dir)
ppo_mean_rewards, ppo_std_rewards, ppo_timesteps = extract_rewards_from_monitors(
    log_dir=ppo_run_dir
)
sac_mean_rewards, sac_std_rewards, sac_timesteps = extract_rewards_from_monitors(
    log_dir=sac_run_dir
)

plot_means_and_stds(
    means_list=[ppo_mean_rewards, shac_mean_rewards, sac_mean_rewards],
    stds_list=[ppo_std_rewards, shac_std_rewards, sac_std_rewards],
    timesteps_list=[ppo_timesteps, shac_timesteps, sac_timesteps],
    series_list=["PPO", "SHAC", "SAC"],
    colors_list=["#FF7700", "#004777", "#B72B34"],
)
