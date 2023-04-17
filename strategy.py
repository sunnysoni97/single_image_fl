from flwr.common import (
    Parameters,
    ndarrays_to_parameters
)
from models import init_model, get_parameters
from typing import Dict, Tuple
from torch.utils.data import DataLoader
import torch
from common import test_model
import numpy as np


class common_functions:
    @staticmethod
    def initialise_parameters(model_name: str, num_classes: int) -> Parameters:
        new_model = init_model(model_name, num_classes)
        parameters = get_parameters(new_model)
        return ndarrays_to_parameters(parameters)

    @staticmethod
    def fit_metrics_aggregation_fn(fit_metrics) -> Dict[str, float]:

        total_clients = 0
        total_train_loss = 0
        total_train_acc = 0

        for metrics in fit_metrics:
            total_clients += 1
            total_train_loss += metrics[1]['train_loss']
            total_train_acc += metrics[1]['train_acc']

        avg_train_loss = total_train_loss/total_clients
        avg_train_acc = total_train_acc/total_clients

        return {'avg_train_loss': avg_train_loss, 'avg_train_acc': avg_train_acc}

    @staticmethod
    def evaluate_metrics_aggregation_fn(eval_metrics) -> Dict[str, float]:

        total_clients = 0
        total_eval_acc = 0

        for metrics in eval_metrics:
            total_clients += 1
            total_eval_acc += metrics[1]['accuracy']

        avg_eval_acc = total_eval_acc / total_clients

        return {'avg_eval_acc': avg_eval_acc}


class fed_avg_fn:
    @staticmethod
    def get_fit_config_fn():
        def on_fit_config_fn(server_round: int) -> Dict[str, float]:
            xp = [x for x in range(1,26)]
            yp = np.linspace(1e-3,1e-5,25)
            lr = np.interp(server_round, xp=xp, fp=yp)
            config = {
                'lr': lr,
                'epochs': 1,
                'momentum': 0.9,
                'round': server_round
            }
            return config
        return on_fit_config_fn

    @staticmethod
    def get_eval_config_fn():
        def on_eval_config_fn(server_round: int) -> Dict[str, float]:
            config = {
                'round': server_round
            }
            return config
        return on_eval_config_fn

    @staticmethod
    def get_evaluate_fn(model_name: str, n_classes: int, test_loader: DataLoader, device: torch.device):
        def evaluate_fn(server_round: int, model_params: Parameters, unk_dict: Dict) -> Tuple[float, Dict[str, float]]:
            test_res = test_model(model_name, n_classes,
                                  model_params, test_loader, device)
            test_loss = test_res['test_loss']
            test_acc = test_res['test_acc']
            return test_loss, {'server_test_acc': test_acc}
        return evaluate_fn
