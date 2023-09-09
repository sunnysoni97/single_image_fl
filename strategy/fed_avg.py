from typing import Dict, Tuple
from torch.utils.data import DataLoader
import torch
from flwr.common import (
    Parameters
)
from flwr.common.logger import log
from logging import INFO

from common import test_model
from strategy.common import cosine_annealing_round


class fed_avg_fn:
    @staticmethod
    def get_fit_config_fn(local_lr: float = 0.005, local_epochs: int = 40, adaptive_lr: bool = False, adaptive_lr_round: bool = False, max_server_rounds: int = 30, use_fedprox: bool = False, fedprox_factor: float = 1.0, use_clipping: bool = False, clipping_factor: float = 2.5):
        def on_fit_config_fn(server_round: int) -> Dict[str, float]:

            lr = local_lr

            if (adaptive_lr_round):
                lr = cosine_annealing_round(
                    max_lr=local_lr, min_lr=1e-3, max_rounds=max_server_rounds, curr_round=server_round)
                log(INFO, f"Current Round Client LR : {lr}")

            config = {
                'lr': lr,
                'epochs': local_epochs,
                'fedprox_factor': fedprox_factor,
                'use_fedprox': use_fedprox,
                'use_adaptive_lr': adaptive_lr,
                'use_clipping': use_clipping,
                'clipping_factor': clipping_factor
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
    def get_evaluate_fn(model_name: str, dataset_name: str, test_loader: DataLoader, device: torch.device):
        def evaluate_fn(server_round: int, model_params: Parameters, unk_dict: Dict) -> Tuple[float, Dict[str, float]]:
            test_res = test_model(model_name, dataset_name,
                                  model_params, test_loader, device)
            test_loss = test_res['test_loss']
            test_acc = test_res['test_acc']
            print(
                f'Server test accuracy after round {server_round}: {test_acc}')
            return test_loss, {'server_test_acc': test_acc}
        return evaluate_fn
