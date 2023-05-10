import flwr as fl
from flwr.server import strategy
import torch
import numpy as np
from torch.utils.data import DataLoader
import random

from data_loader_scripts.download import download_dataset
from data_loader_scripts.partition import do_fl_partitioning
from data_loader_scripts.create_dataloader import combine_val_loaders
from fed_df_data_loader.split_standard import create_std_distill_loader, split_standard
from fed_df_data_loader.get_crops_dataloader import get_distill_imgloader

from strategy.common import common_functions
from strategy.fed_avg import fed_avg_fn
from strategy.fed_df import FedDF_strategy, fed_df_fn

import client.fed_avg
import client.fed_df

from arg_handler import parser

if __name__ == "__main__":

    MODEL_NAME = "resnet18"

    DEVICE = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    args = parser.parse_args()

    FED_STRATEGY = args.fed_strategy

    NUM_CLIENTS = args.num_clients
    NUM_ROUNDS = args.num_rounds
    FRACTION_FIT = args.fraction_fit
    FRACTION_EVALUATE = args.fraction_evaluate

    DATASET_NAME = args.dataset_name
    DATA_DIR = args.data_dir
    PARTITION_ALPHA = args.partition_alpha
    PARTITION_VAL_RATIO = args.partition_val_ratio

    CLIENT_CPUS = args.client_cpus
    CLIENT_GPUS = args.client_gpus
    SERVER_CPUS = args.server_cpus

    BATCH_SIZE = args.batch_size

    LOCAL_EPOCHS = args.local_epochs
    LOCAL_LR = args.local_lr

    SERVER_LR = args.server_lr
    SERVER_STEPS = args.server_steps
    SERVER_EARLY_STEPS = args.server_early_steps
    USE_EARLY_STOPPING = args.use_early_stopping
    USE_ADAPTIVE_LR = args.use_adaptive_lr

    SEED = args.seed
    CUDA_DETERMINISTIC = args.cuda_deterministic

    USE_CROPS = args.use_crops
    DISTILL_DATASET = args.distill_dataset
    NUM_DISTILL_IMAGES = args.num_distill_images

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

    split_test_loaders = split_standard(dataloader=test_loader, alpha=float(
        'inf'), batch_size=BATCH_SIZE, n_workers=SERVER_CPUS, seed=SEED)
    test_loader = split_test_loaders[0]
    distill_dataloader = split_test_loaders[1]

    fed_dir = do_fl_partitioning(
        train_data_path, NUM_CLIENTS, PARTITION_ALPHA, NUM_CLASSES, SEED, PARTITION_VAL_RATIO)

    if(FED_STRATEGY == "fedavg"):
        def client_fn(cid) -> client.fed_avg.FlowerClient:
            return client.fed_avg.FlowerClient(cid, MODEL_NAME, NUM_CLASSES, DATASET_NAME, fed_dir, BATCH_SIZE, CLIENT_CPUS, DEVICE)

    elif(FED_STRATEGY == "feddf"):
        if(USE_CROPS):
            distill_dataloader = get_distill_imgloader(
                f'{DATA_DIR}/single_img_crops/crops', dataset_name=DATASET_NAME, batch_size=BATCH_SIZE, num_workers=CLIENT_CPUS)
        else:
            if(DISTILL_DATASET != DATASET_NAME):
                distill_dataloader = create_std_distill_loader(
                    dataset_name=DISTILL_DATASET, transforms_name=DATASET_NAME, storage_path=DATA_DIR, n_images=NUM_DISTILL_IMAGES, batch_size=BATCH_SIZE, n_workers=SERVER_CPUS, seed=SEED, alpha=100.0)

        val_dataloader = combine_val_loaders(
            dataset_name=DATASET_NAME, path_to_data=fed_dir, n_clients=NUM_CLIENTS, batch_size=BATCH_SIZE, workers=SERVER_CPUS)

        def client_fn(cid) -> client.fed_df.FlowerClient:
            return client.fed_df.FlowerClient(cid=cid, model_name=MODEL_NAME, model_n_classes=NUM_CLASSES, dataset_name=DATASET_NAME, fed_dir=fed_dir, batch_size=BATCH_SIZE, num_cpu_workers=CLIENT_CPUS, device=DEVICE, distill_dataloader=distill_dataloader)

    else:
        raise ValueError(f'{FED_STRATEGY} has not been implemented!')

    client_resources = {"num_cpus": CLIENT_CPUS}
    if (DEVICE.type == "cuda"):
        client_resources["num_gpus"] = CLIENT_GPUS

    if(FED_STRATEGY == "fedavg"):
        # FedAvg
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy.FedAvg(
                on_fit_config_fn=fed_avg_fn.get_fit_config_fn(
                    local_lr=LOCAL_LR, local_epochs=LOCAL_EPOCHS, adaptive_lr_round=USE_ADAPTIVE_LR, max_server_rounds=NUM_ROUNDS),
                on_evaluate_config_fn=fed_avg_fn.get_eval_config_fn(),
                initial_parameters=common_functions.initialise_parameters(
                    MODEL_NAME, NUM_CLASSES),
                fit_metrics_aggregation_fn=common_functions.fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=common_functions.evaluate_metrics_aggregation_fn,
                evaluate_fn=fed_avg_fn.get_evaluate_fn(
                    MODEL_NAME, NUM_CLASSES, test_loader, DEVICE),
                fraction_fit=FRACTION_FIT,
                fraction_evaluate=FRACTION_EVALUATE
            ),
            client_resources=client_resources,
        )

    elif(FED_STRATEGY == "feddf"):
        # FedDF
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=FedDF_strategy(
                fraction_fit=FRACTION_FIT,
                fraction_evaluate=FRACTION_EVALUATE,
                distillation_dataloader=distill_dataloader,
                evaluation_dataloader=test_loader,
                val_dataloader=val_dataloader,
                model_type=MODEL_NAME,
                model_n_classes=NUM_CLASSES,
                device=DEVICE,
                initial_parameters=common_functions.initialise_parameters(
                    MODEL_NAME, NUM_CLASSES),
                fit_metrics_aggregation_fn=common_functions.fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=common_functions.evaluate_metrics_aggregation_fn,
                on_fit_config_fn_client=fed_df_fn.get_on_fit_config_fn_client(
                    client_epochs=LOCAL_EPOCHS, client_lr=LOCAL_LR),
                on_fit_config_fn_server=fed_df_fn.get_on_fit_config_fn_server(
                    server_lr=SERVER_LR, distill_steps=SERVER_STEPS, use_early_stopping=USE_EARLY_STOPPING, early_stop_steps=SERVER_EARLY_STEPS, use_adaptive_lr=USE_ADAPTIVE_LR),
                evaluate_fn=fed_df_fn.evaluate_fn
            ),
            client_resources=client_resources,
        )
