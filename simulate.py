import flwr as fl
from client import FlowerClient
from models import init_model, get_parameters
import torch
from data_loader import load_dataset
from typing import Dict, List
import numpy as np


if __name__ == "__main__":
    NUM_CLIENTS = 2
    NUM_ROUNDS = 10
    MODEL_NAME = "resnet18"
    NUM_CLASSES = 10

    train_loaders, val_loaders, test_loader = load_dataset(
        "cifar10", NUM_CLIENTS, 8, 42)
    DEVICE = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    def client_fn(cid) -> FlowerClient:
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        train_loader = train_loaders[int(cid)]
        val_loader = test_loader
        return FlowerClient(cid, MODEL_NAME, NUM_CLASSES, train_loader, val_loader, device)

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
        return parameters

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=fl.server.strategy.FedAvg(
            on_fit_config_fn=fit_config,
            initial_parameters=initialise_parameters()
        ),
        ray_init_args=ray_init_args,
        client_resources=client_resources,
    )

