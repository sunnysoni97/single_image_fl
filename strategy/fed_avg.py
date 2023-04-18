from typing import Dict, Tuple
from common import test_model
from torch.utils.data import DataLoader
import torch
from flwr.common import (
    Parameters
)

class fed_avg_fn:
    @staticmethod
    def get_fit_config_fn():
        def on_fit_config_fn(server_round: int) -> Dict[str, float]:

            if(server_round < 16):
                lr = 1e-3
            elif(server_round < 21):
                lr = 1e-4
            else:
                lr = 1e-5

            # adaptive lr
            # end_round = 20
            # low = -5
            # high = -2
            # xp = [x for x in range(1,end_round+1)]
            # yp = np.logspace(high,low,end_round)
            # lr = np.interp(server_round, xp=xp, fp=yp)

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