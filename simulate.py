import flwr as fl
from flwr.server import strategy
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import random

from data_loader_scripts.download import download_dataset
from data_loader_scripts.partition import do_fl_partitioning

from strategy.common import common_functions
from strategy.fed_avg import fed_avg_fn

import client.fed_avg

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
parser.add_argument("--cuda_deterministic", type=bool, default=False)


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
    CUDA_DETERMINISTIC = args.cuda_deterministic

    if(DATASET_NAME == "cifar10"):
        NUM_CLASSES = 10
    elif(DATASET_NAME == "cifar100"):
        NUM_CLASSES = 100
    else:
        raise ValueError(f"{DATASET_NAME} has not been implemented yet!")

    # seeding everything

    if(SEED is not None):

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

    if(CUDA_DETERMINISTIC):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    train_data_path, test_set = download_dataset(DATA_DIR, DATASET_NAME)
    kwargs_test_loader = {"num_workers": CLIENT_CPUS,
                          "pin_memory": True, "drop_last": False}
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, **kwargs_test_loader)

    fed_dir = do_fl_partitioning(
        train_data_path, NUM_CLIENTS, PARTITION_ALPHA, NUM_CLASSES, SEED, PARTITION_VAL_RATIO)

    def client_fn(cid) -> client.fed_avg.FlowerClient:
        return client.fed_avg.FlowerClient(cid, MODEL_NAME, NUM_CLASSES, DATASET_NAME, fed_dir, BATCH_SIZE, CLIENT_CPUS, DEVICE)

    client_resources = {"num_cpus": CLIENT_CPUS}
    if (DEVICE.type == "cuda"):
        client_resources["num_gpus"] = CLIENT_GPUS

    # FedAvg
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy.FedAvg(
            on_fit_config_fn=fed_avg_fn.get_fit_config_fn(),
            on_evaluate_config_fn=fed_avg_fn.get_eval_config_fn(),
            initial_parameters=common_functions.initialise_parameters(
                MODEL_NAME, NUM_CLASSES),
            fit_metrics_aggregation_fn=common_functions.fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=common_functions.evaluate_metrics_aggregation_fn,
            evaluate_fn=fed_avg_fn.get_evaluate_fn(
                MODEL_NAME, NUM_CLASSES, test_loader, DEVICE)
        ),
        client_resources=client_resources,
    )
