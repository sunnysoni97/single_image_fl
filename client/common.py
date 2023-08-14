import torch


def fedprox_term(new_wt: torch.tensor, past_wt: torch.tensor, factor: float):
    penalty = factor/2*(torch.sum(new_wt-past_wt))**2
    return penalty
