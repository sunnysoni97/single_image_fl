from flwr.common import (
    ndarrays_to_parameters,
    Parameters
)
from models import init_model, get_parameters
from typing import Dict
import math


class common_functions:
    @staticmethod
    def initialise_parameters(model_name: str, dataset_name: str) -> Parameters:
        new_model = init_model(dataset_name, model_name)
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

        print(f'Average local training loss : {avg_train_loss}')
        print(f'Average local training accuracy : {avg_train_acc}')
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


def cosine_annealing_round(max_lr: float, min_lr: float, max_rounds: int, curr_round: int) -> float:
    min = min_lr
    max = max_lr
    lr = min + (1/2)*(max-min)*(1+math.cos(curr_round/max_rounds*math.pi))
    return lr
