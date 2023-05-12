from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    parameters_to_ndarrays,
    ndarray_to_bytes,
    ndarrays_to_parameters,
)

import flwr as fl
from typing import List, Tuple, Dict
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from pathlib import Path

from models import init_model, set_parameters, get_parameters
from data_loader_scripts.create_dataloader import create_dataloader
from common import test_model


def train_model(model_name: str, model_n_classes: int, parameters: List[np.ndarray], train_loader: DataLoader, distill_loader: DataLoader, config: dict, DEVICE: torch.device, enable_epoch_logging: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
    model = init_model(model_name, model_n_classes)
    set_parameters(model, parameters)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'])
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

            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.detach(), 1)
                        [1] == labels).sum().item()

        epoch_acc = correct/total
        total_epoch_loss.append(epoch_loss)
        total_epoch_acc.append(epoch_acc)
        if((epoch+1) % 10 == 0 and enable_epoch_logging):
            print(f'Epoch {epoch+1} : loss {epoch_loss}, acc {epoch_acc}')

    new_parameters = get_parameters(model)

    model.eval()
    with torch.no_grad():
        distill_preds = []
        for images, _ in distill_loader:
            images = images.to(DEVICE)
            batch_preds = model(images)
            distill_preds.append(batch_preds)

        distill_preds = torch.cat(distill_preds, dim=0).cpu().numpy()

    train_res = {'train_loss': np.mean(
        total_epoch_loss), 'train_acc': np.mean(total_epoch_acc)}

    return (new_parameters, distill_preds, train_res)


class FlowerClient(fl.client.Client):
    def __init__(self, cid: str, model_name: str, model_n_classes: int, dataset_name: str, fed_dir: Path, batch_size: int, num_cpu_workers: int, device: torch.device, distill_dataloader: DataLoader) -> None:
        self.cid = cid
        self.model_name = model_name
        self.model_n_classes = model_n_classes
        self.dataset_name = dataset_name
        self.fed_dir = fed_dir
        self.batch_size = batch_size
        self.num_cpu_workers = num_cpu_workers
        self.device = device
        self.parameters = None
        self.distill_dataloader = distill_dataloader

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print(f'Getting parameters from Client {self.cid}')
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=self.parameters,
        )

    def set_parameters(self, parameters) -> None:
        self.parameters = parameters
        return

    def fit(self, ins: FitIns) -> FitRes:
        print(f'Fitting Client {self.cid}')
        parameters = parameters_to_ndarrays(ins.parameters)
        self.set_parameters(parameters)
        train_loader = create_dataloader(
            self.dataset_name, self.fed_dir, self.cid, True, self.batch_size, self.num_cpu_workers)
        new_parameters, distill_preds, train_res = train_model(
            model_name=self.model_name, model_n_classes=self.model_n_classes, parameters=self.parameters, train_loader=train_loader, distill_loader=self.distill_dataloader, config=ins.config, DEVICE=self.device)

        self.set_parameters(new_parameters)
        train_res['preds'] = ndarray_to_bytes(distill_preds)
        train_res['preds_number'] = len(self.distill_dataloader.dataset)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=ndarrays_to_parameters(new_parameters),
            num_examples=len(train_loader),
            metrics=train_res,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f'Evaluating Client {self.cid}')
        self.set_parameters(parameters_to_ndarrays(ins.parameters))
        val_loader = create_dataloader(
            self.dataset_name, self.fed_dir, self.cid, False, self.batch_size, self.num_cpu_workers)
        val_res = test_model(self.model_name, self.model_n_classes,
                             self.parameters, val_loader, self.device)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(val_res['test_loss']),
            num_examples=len(val_loader),
            metrics={"accuracy": float(val_res['test_acc'])},
        )
