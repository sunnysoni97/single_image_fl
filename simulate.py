import flwr as fl
from flwr.server import strategy
from client import FlowerClient
from models import init_model, get_parameters
import torch
from data_loader import load_dataset
from typing import Dict, List
import numpy as np
import gc


if __name__ == "__main__":
    gc.enable()

    NUM_CLIENTS = 10
    NUM_ROUNDS = 15
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 10
    DEVICE = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_loaders, val_loaders, test_loader = load_dataset(
        "cifar10", NUM_CLIENTS, 8, 42)

    def client_fn(cid) -> FlowerClient:
        train_loader = train_loaders[int(cid)]
        val_loader = test_loader
        return FlowerClient(cid, MODEL_NAME, NUM_CLASSES, train_loader, val_loader, DEVICE)

    def fit_config(server_round: int) -> Dict[str, float]:
        config = {
            'lr': 1e-3,
            'epochs': 1,
            'momentum': 0.9,
        }
        return config

    client_resources = None
    ray_init_args = None
    if (DEVICE.type == "cuda"):
        client_resources = {"num_gpus": 0.5}
        ray_init_args = {"num_gpus": 1}

    def initialise_parameters() -> List[np.ndarray]:
        new_model = init_model(MODEL_NAME, NUM_CLASSES)
        parameters = get_parameters(new_model)
        return fl.common.ndarrays_to_parameters(parameters)

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

    def evaluate_metrics_aggregation_fn(eval_metrics) -> Dict[str, float]:

        total_clients = 0
        total_eval_acc = 0

        for metrics in eval_metrics:
            total_clients += 1
            total_eval_acc += metrics[1]['accuracy']

        avg_eval_acc = total_eval_acc / total_clients

        return {'avg_eval_acc': avg_eval_acc}

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy.FedAvg(
            on_fit_config_fn=fit_config,
            initial_parameters=initialise_parameters(),
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        ),
        ray_init_args=ray_init_args,
        client_resources=client_resources,
    )
