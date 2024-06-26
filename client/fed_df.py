from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
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
import random

from models import init_model, set_parameters, get_parameters, params_to_tensors
from data_loader_scripts.create_dataloader import create_dataloader
from common import test_model
from strategy.tools.clustering import extract_from_transport
from strategy.tools.clipping import clip_logits
from client.common import fedprox_term


def train_model(model_name: str, dataset_name: str, parameters: List[np.ndarray], train_loader: DataLoader, distill_loader: DataLoader, config: dict, DEVICE: torch.device, enable_epoch_logging: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:

    model = init_model(dataset_name, model_name)
    set_parameters(model, parameters)
    criterion = nn.CrossEntropyLoss(reduction="mean")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'])

    if (config['use_adaptive_lr']):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config['epochs'])

    model.train()
    model.to(DEVICE)

    initial_parameters = params_to_tensors(model.parameters()).detach()

    total_epoch_loss = []
    total_epoch_acc = []

    for epoch in range(config['epochs']):
        correct, total, epoch_loss = 0, 0, 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if (config['use_clipping']):
                outputs_temp = model(images)
                outputs = clip_logits(
                    outputs=outputs_temp, scaling_factor=config['clipping_factor'])
            else:
                outputs = outputs_temp

            if (config['use_fedprox']):
                loss_temp = criterion(outputs, labels)
                epoch_loss += loss_temp.item()
                penalty = fedprox_term(new_wt=params_to_tensors(model.parameters()),
                                       past_wt=initial_parameters, factor=config['fedprox_factor'])
                loss = loss_temp + penalty

            else:
                loss = criterion(outputs, labels)
                epoch_loss += loss.item()

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
    def __init__(self, cid: str, model_name: str, dataset_name: str, fed_dir: Path, batch_size: int, num_cpu_workers: int, device: torch.device, debug: bool = False, seed: int = None, cuda_deterministic: bool = False) -> None:
        self.cid = cid
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.fed_dir = fed_dir
        self.batch_size = batch_size
        self.num_cpu_workers = num_cpu_workers
        self.device = device
        self.parameters = None
        self.debug = debug

        if (seed is not None):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        if (cuda_deterministic):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        result_dict = {}
        status_code = 404
        for key in ins.config.keys():
            if (hasattr(self, key)):
                result_dict[key] = getattr(self, key)
                status_code = 200
            else:
                result_dict[key] = "None"

        res = GetPropertiesRes(status=status_code, properties=result_dict)
        return res

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
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

        distill_loader = extract_from_transport(
            img_bytes=ins.config['distill_crops'], batch_size=self.batch_size, n_workers=self.num_cpu_workers)

        new_parameters, distill_preds, train_res = train_model(
            model_name=self.model_name, dataset_name=self.dataset_name, parameters=self.parameters, train_loader=train_loader, distill_loader=distill_loader, config=ins.config, DEVICE=self.device, enable_epoch_logging=self.debug)

        self.set_parameters(new_parameters)
        train_res['preds'] = ndarray_to_bytes(distill_preds)
        train_res['preds_number'] = len(distill_loader)

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
        val_res = test_model(self.model_name, self.dataset_name,
                             self.parameters, val_loader, self.device)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(val_res['test_loss']),
            num_examples=len(val_loader),
            metrics={"accuracy": float(val_res['test_acc'])},
        )
