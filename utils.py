from torch.nn import Module


def hard_network_update(net: Module, targ_net: Module):
    """
    Updates the network parameters to be like the target network parameters
    """
    params = net.parameters()
    target_params = targ_net.parameters()

    for param, target_param in zip(params, target_params):
        param.data.copy_(target_param.data)
