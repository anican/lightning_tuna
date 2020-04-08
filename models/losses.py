import torch
import torch.nn.functional as F


def linear_hinge_loss(output, target, reduction='mean') -> float:
    binary_target = output.new_empty(*output.shape).fill_(-1)
    for i in range(len(target)):
        binary_target[i, target[i]] = 1
    delta = 1 - binary_target * output
    delta[delta <= 0] = 0
    return delta.mean() if reduction == 'mean' else delta.sum()


def p_loss(output, target, p) -> float:
    loss = torch.mean(torch.sum((torch.abs(output-target)**p),
        dim=1)**(1./p)).item()
    return loss


def square_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
