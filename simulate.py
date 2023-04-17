import flwr as fl
from flwr.server import strategy
from client import FlowerClient
from models import init_model, get_parameters
import torch
from typing import Dict, List
import numpy as np
import argparse
from data_loader_scripts.download import download_dataset
from data_loader_scripts.partition import do_fl_partitioning

parser = argparse.ArgumentParser(description="FedAvg Simulation using Flower")

parser.add_argument("--num_clients", type=int, default=10)
parser.add_argument("--num_rounds", type=int, default=10)

parser.add_argument("--dataset_name", type=str, default="cifar10")
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--partition_alpha", type=float, default=1000.0)
parser.add_argument("--partition_val_ratio", type=float, default=0.1)

parser.add_argument("--client_cpus", type=int, default=2)
parser.add_argument("--client_gpus", type=float, default=0.5)

parser.add_argument("--seed", type=int, default=None)


if __name__ == "__main__":

    MODEL_NAME = "resnet18"
    BATCH_SIZE = 32
    DEVICE = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    args = parser.parse_args()

    NUM_CLIENTS = args.num_clients
    NUM_ROUNDS = args.num_rounds
    DATASET_NAME = args.dataset_name
    DATA_DIR = args.data_dir
    PARTITION_ALPHA = args.partition_alpha
    PARTITION_VAL_RATIO = args.partition_val_ratio
    CLIENT_CPUS = args.client_cpus
    CLIENT_GPUS = args.client_gpus
    SEED = args.seed

    if(DATASET_NAME == "cifar10"):
        NUM_CLASSES = 10
    elif(DATASET_NAME == "cifar100"):
        NUM_CLASSES = 100
    else:
        raise ValueError(f"{DATASET_NAME} has not been implemented yet!")

    train_data_path, test_loader = download_dataset(DATA_DIR, DATASET_NAME)

    fed_dir = do_fl_partitioning(
        train_data_path, NUM_CLIENTS, PARTITION_ALPHA, NUM_CLASSES, SEED, PARTITION_VAL_RATIO)

    def client_fn(cid) -> FlowerClient:
        return FlowerClient(cid, MODEL_NAME, NUM_CLASSES, DATASET_NAME, fed_dir, BATCH_SIZE, CLIENT_CPUS, DEVICE)

    def fit_config(server_round: int) -> Dict[str, float]:
        config = {
            'lr': 1e-3,
            'epochs': 1,
            'momentum': 0.9,
        }
        return config

    client_resources = {"num_cpus": CLIENT_CPUS}
    if (DEVICE.type == "cuda"):
        client_resources["num_gpus"] = CLIENT_GPUS

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
        client_resources=client_resources,
    )
