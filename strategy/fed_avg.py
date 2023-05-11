from typing import Dict, Tuple
from common import test_model
from torch.utils.data import DataLoader
import torch
import numpy as np
from flwr.common import (
    Parameters
)


class fed_avg_fn:
    @staticmethod
    def get_fit_config_fn(local_lr: float = 0.005, local_epochs: int = 40, adaptive_lr_round: bool = False, max_server_rounds: int = 30):
        def on_fit_config_fn(server_round: int) -> Dict[str, float]:

            lr = local_lr

            if(adaptive_lr_round):
                xp = [x for x in range(1, max_server_rounds+1)]
                yp = np.logspace(local_lr, 0.0, max_server_rounds)
                lr = np.interp(server_round, xp=xp, fp=yp)

            config = {
                'lr': lr,
                'epochs': local_epochs,
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
