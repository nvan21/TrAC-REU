from torch.nn import ELU
from torch import device, cuda


class Config:
    """
    Base class that each config inherits from. These will provide default values for each task, but they can be
    overwritten by changing the value in the new task class.
    """

    # Default configurations for the actor neural network
    actor_units = [64, 64]
    actor_activation = ELU()
    actor_learning_rate = 2e-3
    actor_learning_rate_schedule = "linear"  # can be linear or constant

    # Default configurations for the critic neural network
    critic_units = [64, 64]
    critic_activation = ELU()
    critic_learning_rate = 2e-4
    critic_learning_rate_schedule = "linear"  # can be linear or constant

    # Default hyperparameters for training
    gae_lambda = 0.95
    gamma = 0.99
    num_steps = 32  # this is the length of the trajectory (h in the paper)
    num_envs = 64  # this is the number of parallel envs (N in the paper)
    max_epochs = 2000

    # Device to use for tensor storage
    device = device("cuda" if cuda.is_available() else "cpu")


class HopperConfig(Config):
    actor_units = [128, 128]

    num_envs = 4
    num_steps = 16
