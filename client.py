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


def train_model(model: ResNet, train_loader: DataLoader, config: dict, DEVICE: torch.device) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.SGD(params=model.parameters(
    ), lr=config['lr'], momentum=config['momentum'])
    model.train()

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

    train_res = {'train_loss': np.mean(
        total_epoch_loss), 'train_acc': np.mean(total_epoch_acc)}
    return train_res


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, model: ResNet, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> None:
        self.cid = cid
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def get_parameters(self, config) -> List[np.ndarray]:
        print(f'Getting parameters from Client {self.cid}')
        return get_parameters(self.model)

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        print(f'Fitting Client {self.cid}')
        set_parameters(self.model, parameters)
        train_res = train_model(
            self.model, self.train_loader, config, self.device)
        return (self.get_parameters(config), len(self.train_loader), train_res)

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, float]]:
        print(f'Evaluating Client {self.cid}')
        set_parameters(self.model, parameters)
        val_res = test_model(self.model, self.val_loader, self.device)
        return (val_res['test_loss'], len(self.val_loader), {"accuracy": val_res['test_acc']})
