import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gymnasium.envs.classic_control.pendulum import PendulumEnv
import pickle
from stable_baselines3 import PPO

from utils.hyperparameters import PPOPendulumParams, Experiments
from utils.utils import set_seed


# Main Training Script
if __name__ == "__main__":
    env_id = "Pendulum-v1"
    params = PPOPendulumParams()
    seeds = Experiments.seeds  # List of seeds
    num_envs = params.num_envs  # Number of parallel environments

    for seed in seeds:
        # Create log directory if it doesn't exist already
        log_dir = f"../experiments/length_parameter_curves/PPO/seed_{seed}"
        os.makedirs(log_dir, exist_ok=True)

        # Create environment
        env = PendulumEnv()

        # Load the PPO model
        model = PPO.load(f"../experiments/training_curves/PPO/seed_{seed}/best_policy")
        model.set_random_seed(seed=seed)
        model.policy.eval()

        # Set the PRNG seed
        set_seed(seed=seed)
        rewards = []
        lengths = []

        for length in Experiments.lengths:
            env.l = length
            state, _ = env.reset(seed=seed)
            done, truncated = False, False
            total_reward = 0

            for i in range(200):
                action, _ = model.predict(state)
                state, reward, done, truncated, _ = env.step(action)

                total_reward += reward

            print(f"Final reward for pendulum length of {length}: {total_reward}")
            rewards.append(total_reward)
            lengths.append(length)

        with open(f"{log_dir}/data.pkl", "wb") as f:
            data = {"rewards": rewards, "lengths": lengths}
            pickle.dump(data, f)
