from torchvision.models import ResNet
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from typing import Dict, List, Tuple
import numpy as np
import flwr as fl
from models import get_parameters, set_parameters, init_model
from common import test_model

# function for training a model


def train_model(model_name: str, model_n_classes: int, parameters: List[np.ndarray], train_loader: DataLoader, config: dict, DEVICE: torch.device) -> Tuple[List[np.ndarray], Dict[str, float]]:
    model = init_model(model_name, model_n_classes)
    set_parameters(model, parameters)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.SGD(params=model.parameters(
    ), lr=config['lr'], momentum=config['momentum'])
    model.train()
    model.to(DEVICE)

    total_epoch_loss = []
    total_epoch_acc = []

    for epoch in range(config['epochs']):
        correct, total, epoch_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        epoch_loss /= len(train_loader.dataset)
        epoch_acc = correct/total
        total_epoch_loss.append(epoch_loss.item())
        total_epoch_acc.append(epoch_acc)
        print(f'Epoch {epoch+1} : loss {epoch_loss}, acc {epoch_acc}')

    new_parameters = get_parameters(model)
    train_res = {'train_loss': np.mean(
        total_epoch_loss), 'train_acc': np.mean(total_epoch_acc)}
    del model
    torch.cuda.empty_cache()
    return (new_parameters, train_res)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, model_name: str, model_n_classes: int, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> None:
        self.cid = cid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_name = model_name
        self.model_n_classes = model_n_classes
        self.parameters = None

    def get_parameters(self, config) -> List[np.ndarray]:
        print(f'Getting parameters from Client {self.cid}')
        return self.parameters

    def set_parameters(self, parameters) -> None:
        self.parameters = parameters
        return

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        print(f'Fitting Client {self.cid}')
        self.set_parameters(parameters)
        new_params, train_res = train_model(
            self.model_name, self.model_n_classes, self.parameters, self.train_loader, config, self.device)
        self.set_parameters(new_params)
        return (self.get_parameters(config), len(self.train_loader), train_res)

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, float]]:
        print(f'Evaluating Client {self.cid}')
        self.set_parameters(parameters)
        val_res = test_model(self.model_name, self.model_n_classes,
                             self.parameters, self.val_loader, self.device)
        return (val_res['test_loss'], len(self.val_loader), {"accuracy": val_res['test_acc']})
