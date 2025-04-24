import ray
import torch

@ray.remote
def distributed_l2_loss(input, target, mask, batch_size):
    """Distributed L2 loss computation"""
    loss = (input - target) * mask
    loss = (loss * loss) / 2 / batch_size
    return loss.sum()

# Keep original function for compatibility
l2_loss = distributed_l2_loss
