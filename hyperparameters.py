from torch.nn import ELU, ReLU
from torch import device, cuda


class HopperParams:
    # Project name for wandb
    project_name = "shac-hopper"

    # Configurations for the actor neural network
    actor_units = [128, 128]
    actor_activation = ELU()
    actor_learning_rate = 2e-3
    actor_learning_rate_schedule = "linear"  # can be linear or constant

    # Configurations for the critic neural network
    critic_units = [64, 64]
    critic_activation = ELU()
    critic_learning_rate = 2e-4
    critic_learning_rate_schedule = "linear"  # can be linear or constant
    critic_minibatches = 4

    # Hyperparameters for training
    gae_lambda = 0.95
    gamma = 0.99
    tau = 0.2
    num_steps = 32  # this is the length of the trajectory (h in the paper)
    num_envs = 64  # this is the number of parallel envs (N in the paper)
    max_epochs = 2000
    critic_iterations = 16

    # Whether or not to log run with wandb (useful for debugging)
    do_wandb_logging = False

    # Device to use for tensor storage
    device = device("cuda" if cuda.is_available() else "cpu")


class PendulumParams:
    # Project name for wandb
    project_name = "shac-pendulum"

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
    max_epochs = 2000
    critic_iterations = 16

    # Whether or not to log run with wandb (useful for debugging)
    do_wandb_logging = False

    # Device to use for tensor storage
    device = device("cuda" if cuda.is_available() else "cpu")
