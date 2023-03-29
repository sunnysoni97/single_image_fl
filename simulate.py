import flwr as fl
from client import FlowerClient
from models import init_model
import torch
from data_loader import load_dataset
from typing import Dict


if __name__ == "__main__":
    NUM_CLIENTS = 2
    train_loaders, val_loaders, test_loader = load_dataset(
        "cifar10", NUM_CLIENTS, 8, 42)
    DEVICE = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    def client_fn(cid) -> FlowerClient:
        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = init_model("resnet18", 10)
        model.to(device)
        train_loader = train_loaders[int(cid)]
        val_loader = test_loader
        return FlowerClient(cid, model, train_loader, val_loader, device)

    def fit_config(server_round: int) -> Dict[str, float]:
        config = {
            'lr': 1e-3,
            'epochs': 1,
            'momentum': 0.9,
        }
        return config

    client_resources = None
    if (DEVICE.type == "cuda"):
        client_resources = {"num_gpus": 1}

    fl.server.strategy.FedAvg()

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=fl.server.strategy.FedAvg(
            on_fit_config_fn=fit_config
        ),
        client_resources=client_resources
    )
