from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet
import numpy as np
from typing import List, Union
import torch
import re

from models.cifar_resnet import cifar_resnet, CifarResNet
from models.cifar_wrn import get_wrn_cifar, CIFARWRN

# helper functions


def extract_int(inp_str: str) -> int:
    num = re.findall(r'\d+', inp_str)
    if (len(num) == 0):
        raise ValueError(f'{inp_str} is an incorrect argument!')
    return num

# function for initialising and returning a model


def init_model(dataset_name: str, model_name: str) -> Union[ResNet, CifarResNet, CIFARWRN]:

    if (not (model_name.startswith("resnet") or model_name.startswith("wresnet"))):
        raise ValueError(f"{model_name} not implemented yet!")

    if (dataset_name in ["cifar10", "cifar100", "pathmnist", "pneumoniamnist", "organamnist"]):
        if (model_name.startswith("resnet")):
            variant = int(extract_int(model_name)[0])
            model = cifar_resnet(dataset_name, variant)
        else:
            num_list = extract_int(model_name)
            depth = int(num_list[0])
            width = int(num_list[1])

            if (dataset_name == "cifar10"):
                num_classes = 10
            elif (dataset_name == "cifar100"):
                num_classes = 100
            elif (dataset_name == "pathmnist"):
                num_classes = 9
            else:
                raise ValueError(
                    "Incorrect dataset name for cifar wide resnet!")

            model = get_wrn_cifar(
                num_classes=num_classes, blocks=depth, width_factor=width, model_name=model_name)

    else:
        torch_models = {"resnet18": resnet18,
                        "resnet34": resnet34, "resnet50": resnet50}
        if (model_name in torch_models.keys()):
            model = torch_models[model_name](weights=None, progress=False)
        else:
            raise ValueError(f'{model_name} not implemented for ImageNet!')

    return model

# function for retrieving parameters of a network


def get_parameters(model: Union[ResNet, CifarResNet, CIFARWRN]) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# functions for overwriting paramaters of a network


def set_parameters(model: Union[ResNet, CifarResNet, CIFARWRN], parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {}
    for k, v in params_dict:
        if (v.shape == ()):
            state_dict[k] = torch.Tensor([v.item()])
        else:
            state_dict[k] = torch.Tensor(v)
    model.load_state_dict(state_dict, strict=True)

# functions for retrieving parameters as tensors


def params_to_tensors(param_iterator) -> torch.Tensor:
    params = []
    for param in param_iterator:
        params.append(param.flatten())

    params = torch.cat(tensors=params)
    return params


# DEBUG FUNCTIONS

def print_bn_values(model: Union[ResNet, CifarResNet, CIFARWRN], n_outputs: int = 4):
    i = 0
    for name, values in model.named_parameters():
        if (name.find("bn") > -1 and i < n_outputs):
            print(name)
            print(values)
            i += 1
