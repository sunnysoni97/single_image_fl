from torchvision.models import resnet18
from torchvision.models import ResNet
import torch.nn as nn


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
