import torch


def clip_logits(outputs: torch.Tensor, scaling_factor: float = 1.0) -> torch.Tensor:
    # outputs: B x K
    tau = 1/scaling_factor

    # first get norms
    norms = torch.norm(outputs, p=2, dim=1, keepdim=True)  # B x 1
    normalised_outputs = (outputs / norms) * scaling_factor   # B x K

    # create a mask
    to_clip = norms >= tau  # B x 1

    # add masked logits + unmasked original logits
    return to_clip * normalised_outputs +\
        (~to_clip) * outputs
