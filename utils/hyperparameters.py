from torch.nn import ELU, ReLU
from torch import device, cuda
import numpy as np


class Experiments:
    # Experiment seeds
    seeds = [0, 100, 200, 300, 400]

    # Different experiments
    noises = np.linspace(0, 0.2, 100).tolist()
    masses = np.linspace(0.75, 1.25, 100).tolist()
    lengths = np.linspace(0.2, 1.8, 100).tolist()


class SHACPendulumParams:
    # Project name for wandb
    project_name = "shac-pendulum"

    # Flag for training or testing
    do_train = True

    # Whether or not to log run with wandb (useful for debugging)
    do_wandb_logging = False

    # Configurations for the actor neural network
    actor_units = [64, 64]
    actor_activation = ReLU()
    actor_learning_rate = 1e-3
    actor_learning_rate_schedule = "linear"  # can be linear or constant

    # Configurations for the critic neural network
    critic_units = [64, 64]
    critic_activation = ReLU()
    critic_learning_rate = 1e-3
    critic_learning_rate_schedule = "linear"  # can be linear or constant
    critic_minibatches = 4

    # Hyperparameters for training
    gae_lambda = 0.95
    gamma = 0.99
    tau = 0.1
    num_steps = 32  # this is the length of the trajectory (h in the paper)
    num_envs = 64  # this is the number of parallel envs (N in the paper)
    max_timesteps = 5e5
    critic_iterations = 16

    # Device to use for tensor storage
    device = device("cuda" if cuda.is_available() else "cpu")


class PPOPendulumParams:
    # Project name for wandb
    project_name = "ppo-pendulum"

    # Whether or not to log run with wandb (useful for debugging)
    do_wandb_logging = False

    # Hyperparameters for training
    gae_lambda = 0.95
    gamma = 0.99
    tau = 0.1
    num_steps = 32  # this is the length of the trajectory (h in the paper)
    num_envs = 4  # this is the number of parallel envs (N in the paper)
    max_timesteps = 5e5
    learning_rate = 1e3

    # Device to use for tensor storage
    device = device("cuda" if cuda.is_available() else "cpu")


class SACPendulumParams:
    # Project name for wandb
    project_name = "sac-pendulum"

    # Whether or not to log run with wandb (useful for debugging)
    do_wandb_logging = False

    # Hyperparameters for training
    gae_lambda = 0.95
    gamma = 0.99
    tau = 0.1
    num_steps = 32  # this is the length of the trajectory (h in the paper)
    num_envs = 4  # this is the number of parallel envs (N in the paper)
    max_timesteps = 5e5
    learning_rate = 1e3
    buffer_size = 1e6
    batch_size = 512

    # Device to use for tensor storage
    device = device("cuda" if cuda.is_available() else "cpu")
