import torch

def clip_logits(outputs:torch.Tensor, scaling_factor:float=1.0) -> torch.Tensor:
    
    device = outputs.device
    tou = 1/scaling_factor
    split_outputs = torch.split(outputs,1,dim=0)

    def clip_single_logit(output:torch.Tensor) -> torch.Tensor:
        norm = torch.norm(output, p=2).to(device)
        if(norm >= tou):
            return scaling_factor*output/norm
        else:
            return output
    
    clipped_logits = torch.vstack([clip_single_logit(x) for x in split_outputs])
    return clipped_logits