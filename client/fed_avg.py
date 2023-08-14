from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from typing import Dict, List, Tuple
import numpy as np
import flwr as fl
from pathlib import Path

from models import get_parameters, set_parameters, init_model, params_to_tensors
from common import test_model
from data_loader_scripts.create_dataloader import create_dataloader
from client.common import fedprox_term

# function for training a model


def train_model(model_name: str, dataset_name: str, parameters: List[np.ndarray], train_loader: DataLoader, config: dict, DEVICE: torch.device, enable_epoch_logging: bool = False) -> Tuple[List[np.ndarray], Dict[str, float]]:
    model = init_model(dataset_name, model_name)
    set_parameters(model, parameters)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'])

    if (config['use_adaptive_lr']):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config['epochs'])

    model.train()
    model.to(DEVICE)

    initial_parameters = params_to_tensors(model.parameters())

    total_epoch_loss = []
    total_epoch_acc = []

    for epoch in range(config['epochs']):
        correct, total, epoch_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            if (config['use_fedprox']):
                loss += fedprox_term(new_wt=params_to_tensors(model.parameters()),
                                     past_wt=initial_parameters, factor=config['fedprox_factor'])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total += labels.size(0)
            correct += (torch.max(outputs.detach(), 1)
                        [1] == labels).sum().item()

        if (config['use_adaptive_lr']):
            scheduler.step()

        epoch_acc = correct/total
        total_epoch_loss.append(epoch_loss)
        total_epoch_acc.append(epoch_acc)
        if ((epoch+1) % 10 == 0 and enable_epoch_logging):
            print(f'Epoch {epoch+1} : loss {epoch_loss}, acc {epoch_acc}')

    new_parameters = get_parameters(model)
    train_res = {'train_loss': np.mean(
        total_epoch_loss), 'train_acc': np.mean(total_epoch_acc)}
    return (new_parameters, train_res)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, model_name: str, dataset_name: str, fed_dir: Path, batch_size: int, num_cpu_workers: int, device: torch.device) -> None:
        self.cid = cid
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.fed_dir = fed_dir
        self.batch_size = batch_size
        self.num_cpu_workers = num_cpu_workers
        self.device = device
        self.parameters = None

    def get_parameters(self, config) -> List[np.ndarray]:
        return self.parameters

    def set_parameters(self, parameters) -> None:
        self.parameters = parameters
        return

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        print(f'Fitting Client {self.cid}')
        self.set_parameters(parameters)
        train_loader = create_dataloader(
            self.dataset_name, self.fed_dir, self.cid, True, self.batch_size, self.num_cpu_workers)
        new_params, train_res = train_model(
            self.model_name, self.dataset_name, self.parameters, train_loader, config, self.device)
        self.set_parameters(new_params)
        return (self.get_parameters(config), len(train_loader), train_res)

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict[str, float]]:
        print(f'Evaluating Client {self.cid}')
        self.set_parameters(parameters)
        val_loader = create_dataloader(
            self.dataset_name, self.fed_dir, self.cid, False, self.batch_size, self.num_cpu_workers)
        val_res = test_model(self.model_name, self.dataset_name,
                             self.parameters, val_loader, self.device)
        return (val_res['test_loss'], len(val_loader), {"accuracy": val_res['test_acc']})
