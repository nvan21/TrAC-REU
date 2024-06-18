from torch.nn import Module
import torch
import numpy as np


def hard_network_update(net: Module, targ_net: Module):
    """
    Updates the network parameters to be like the target network parameters
    """
    params = net.parameters()
    target_params = targ_net.parameters()

    for param, target_param in zip(params, target_params):
        param.data.copy_(target_param.data)


def np_to_tensor(
    device: torch.device, arrays: list, dtype=torch.float32
) -> torch.Tensor:
    """
    Converts a list of numpy arrays into torch tensors
    """
    tensors = []

    for array in arrays:
        if isinstance(array, np.ndarray):
            tensors.append(torch.tensor(array, dtype=dtype).to(device))

    return tensors
