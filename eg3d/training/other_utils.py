import json
import numpy as np
import torch
from ipdb import set_trace as st

def convert_to_serializable(obj):
    """Convert PyTorch tensors to NumPy arrays for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    return obj

def get_optimizer_parameters(optimizer):
    """
    Get learning parameters and their learning rates from an optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer.

    Returns:
        dict: A dictionary containing parameter names and their corresponding learning rates.
    """
    optimizer_params = {}
    for group in optimizer.param_groups:
        # for param in group['params']:
        name = group['name']
        optimizer_params[name] = group['lr']
        
    # Convert tensors to serializable format (NumPy arrays)
    optimizer_params = {k: convert_to_serializable(v) for k, v in optimizer_params.items()}
    # print(optimizer_params)

    return optimizer_params
