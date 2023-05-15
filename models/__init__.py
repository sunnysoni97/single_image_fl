from torchvision.models import resnet18
from torchvision.models import ResNet
import torch.nn as nn
import numpy as np
from typing import List
import torch
import torch.nn as nn

# function for initialising and returning a model


def init_model(model_name: str, n_classes: int) -> ResNet:

    # loading standard model from pytorch with random initialisation
    if(model_name == "resnet18"):
        model = resnet18(weights=None, progress=False)
    else:
        raise ValueError(f'{model_name} model has not been defined yet!')

    # changing output layer to match the number of classes required
    if(n_classes < 0):
        raise ValueError(
            f'{n_classes} is not a valid number of output classes!')
    model.fc = nn.Linear(in_features=model.fc.in_features,
                         out_features=n_classes, bias=True)
    return model

# function for retrieving parameters of a network


def get_parameters(model: ResNet) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# functions for overwriting paramaters of a network


def set_parameters(model: ResNet, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {}
    for k, v in params_dict:
        if(v.shape == ()):
            state_dict[k] = torch.Tensor([v.item()])
        else:
            state_dict[k] = torch.Tensor(v)
    model.load_state_dict(state_dict, strict=True)


# DEBUG FUNCTIONS

def print_bn_values(model:ResNet,n_outputs:int=4):
    i=0
    for name,values in model.named_parameters():
        if(name.find("bn")>-1 and i<n_outputs):
            print(name)
            print(values)
            i+=1

