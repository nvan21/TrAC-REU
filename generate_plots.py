import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import pandas as pd


def extract_pickle_data(log_dir, data_id):
    seeds = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]
    rewards = []
    for seed in seeds:
        with open(os.path.join(log_dir, seed, "data.pkl"), "rb") as f:
            data = pickle.load(f)
            rewards.append(data["rewards"])
            timesteps = data[data_id]

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
    means_list,
    stds_list,
    timesteps_list,
    series_list,
    title,
    xaxis,
    yaxis,
    file_name,
    save_dir,
):
    plt.figure(figsize=(12, 8))
    for idx, (means, stds) in enumerate(zip(means_list, stds_list)):
        ts = timesteps_list[idx]
        label = series_list[idx]
        plt.plot(ts, means, label=label, linewidth=1.5)
        plt.fill_between(ts, means - stds, means + stds, alpha=0.3)

    plt.title(title, fontsize=20, fontweight="bold")
    plt.xlabel(xaxis, fontsize=14)
    plt.ylabel(yaxis, fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{file_name}.png", format="png")
    plt.savefig(f"{save_dir}/{file_name}.svg", format="svg")


if __name__ == "__main__":

    curves = [
        "mass_parameter_curves",
        "noise_curves",
        "training_curves",
        "length_parameter_curves",
    ]
    models = ["SHAC", "PPO", "SAC"]
    data_ids = ["masses", "noises", "timesteps", "lengths"]
    titles = [
        "Pendulum Mass Robustness Comparison",
        "Pendulum Noise Robustness Comparison",
        "Pendulum Learning Comparison",
        "Pendulum Length Robustness Comparison",
    ]
    xlabels = [
        "Pendulum mass (kg)",
        "Noise standard deviation",
        f"Simulation steps (x10\u2075)",
        "Pendulum length (m)",
    ]
    ylabels = ["Reward", "Reward", "Reward", "Reward"]
    file_names = [
        "Mass Robustness Comparison",
        "Noise Robustness Comparison",
        "Learning Curve Comparison",
        "Length Robustness Comparison",
    ]

    for i, curve in enumerate(curves):
        means_list, stds_list, timesteps_list = [], [], []
        for model in models:
            log_dir = os.path.join("experiments", curve, model)
            try:
                mean_reward, std_reward, timesteps = extract_pickle_data(
                    log_dir=log_dir, data_id=data_ids[i]
                )
            except FileNotFoundError:
                mean_reward, std_reward, timesteps = extract_rewards_from_monitors(
                    log_dir=log_dir
                )
            except Exception as e:
                print("An error occurred:", e)

            means_list.append(mean_reward)
            stds_list.append(std_reward)
            timesteps_list.append(timesteps)

        fig_dir = os.path.join("experiments", curve)
        plot_means_and_stds(
            means_list=means_list,
            stds_list=stds_list,
            timesteps_list=timesteps_list,
            series_list=models,
            title=titles[i],
            xaxis=xlabels[i],
            yaxis=ylabels[i],
            file_name=file_names[i],
            save_dir=fig_dir,
        )
