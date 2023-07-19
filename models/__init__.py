from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet
import torch.nn as nn
import numpy as np
from typing import List, Union
import torch
import torch.nn as nn
import re
from models.cifar_resnet import cifar_resnet, CifarResNet

# helper functions


def extract_int(inp_str: str) -> int:
    num = re.findall(r'\d+$', inp_str)
    if (len(num) == 0):
        raise ValueError(f'{inp_str} is an incorrect argument!')
    num = int(num[0])
    return num

# function for initialising and returning a model


def init_model(dataset_name: str, model_name: str) -> Union[ResNet, CifarResNet]:

    if (not (model_name.startswith("resnet"))):
        raise ValueError(f"{model_name} not implemented yet!")

    if (dataset_name in ["cifar10", "cifar100", "pathmnist", "pneumoniamnist", "organamnist"]):
        variant = extract_int(model_name)
        model = cifar_resnet(dataset_name, variant)

    else:
        torch_models = {"resnet18": resnet18,
                        "resnet34": resnet34, "resnet50": resnet50}
        if (model_name in torch_models.keys()):
            model = torch_models[model_name](weights=None, progress=False)
        else:
            raise ValueError(f'{model_name} not implemented for ImageNet!')

    return model

# function for retrieving parameters of a network


def get_parameters(model: ResNet) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# functions for overwriting paramaters of a network


def set_parameters(model: ResNet, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {}
    for k, v in params_dict:
        if (v.shape == ()):
            state_dict[k] = torch.Tensor([v.item()])
        else:
            state_dict[k] = torch.Tensor(v)
    model.load_state_dict(state_dict, strict=True)


# DEBUG FUNCTIONS

def print_bn_values(model: ResNet, n_outputs: int = 4):
    i = 0
    for name, values in model.named_parameters():
        if (name.find("bn") > -1 and i < n_outputs):
            print(name)
            print(values)
            i += 1
