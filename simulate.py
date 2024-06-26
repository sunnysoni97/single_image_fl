import flwr as fl
from flwr.server import strategy
import torch
import numpy as np
from torch.utils.data import DataLoader
import random
import os
import time
import logging
import sys

from data_loader_scripts.download import download_dataset
from data_loader_scripts.partition import do_fl_partitioning
from data_loader_scripts.create_dataloader import combine_val_loaders
from fed_df_data_loader.split_standard import create_std_distill_loader, split_standard
from fed_df_data_loader.get_crops_dataloader import get_distill_imgloader

from strategy.common import common_functions
from strategy.fed_avg import fed_avg_fn
from strategy.fed_df import FedDF_strategy, fed_df_fn
from strategy.fed_df_hetero import FedDF_hetero_strategy

import client.fed_avg
import client.fed_df

from arg_handler import parser

# disabling tqdm percentages in download

from torch.utils.model_zoo import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

if __name__ == "__main__":

    # initialise same logger as flower
    logger = logging.getLogger("simulate")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.DEBUG)
    DEFAULT_FORMATTER = logging.Formatter(
        "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
    )
    handler.setFormatter(DEFAULT_FORMATTER)
    logger.addHandler(handler)
    log = logger.log
    INFO = logging.INFO
    DEBUG = logging.DEBUG

    log(INFO, f"Python script started : {__file__.split('/')[-1]}")

    DEVICE = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    log(DEBUG, "Reading args from argument parser")

    args = parser.parse_args()

    FED_STRATEGY = args.fed_strategy
    MODEL_NAME = args.model_name
    MODEL_LIST = args.model_list

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
    TOTAL_CPUS = args.total_cpus
    TOTAL_GPUS = args.total_gpus
    TOTAL_MEM = args.total_mem

    BATCH_SIZE = args.batch_size

    LOCAL_EPOCHS = args.local_epochs
    LOCAL_LR = args.local_lr

    DISTILL_BATCH_SIZE = args.distill_batch_size
    SERVER_LR = args.server_lr
    SERVER_STEPS = args.server_steps
    SERVER_STEPS_ADAPTIVE = args.server_steps_adaptive
    SERVER_STEPS_ADAPTIVE_MIN = args.server_steps_adaptive_min
    SERVER_STEPS_ADAPTIVE_INTERVAL = args.server_steps_adaptive_interval
    SERVER_EARLY_STEPS = args.server_early_steps
    USE_EARLY_STOPPING = args.use_early_stopping
    USE_ADAPTIVE_LR = args.use_adaptive_lr
    USE_ADAPTIVE_LR_ROUND = args.use_adaptive_lr_round

    SEED = args.seed
    CUDA_DETERMINISTIC = args.cuda_deterministic

    USE_CROPS = args.use_crops
    DISTILL_DATASET = args.distill_dataset
    DISTILL_ALPHA = args.distill_alpha
    NUM_DISTILL_IMAGES = args.num_distill_images
    NUM_TOTAL_IMAGES = args.num_total_images
    DISTILL_TRANSFORMS = args.distill_transforms

    WARM_START = args.warm_start
    WARM_START_ROUNDS = args.warm_start_rounds
    WARM_START_INTERVAL = args.warm_start_interval

    KMEANS_N_CLUSTERS = args.kmeans_n_clusters
    KMEANS_HEURISTICS = args.kmeans_heuristics
    KMEANS_MIXED_FACTOR = args.kmeans_mixed_factor
    KMEANS_BALANCING = args.kmeans_balancing

    CONFIDENCE_THRESHOLD = args.confidence_threshold
    CONFIDENCE_STRATEGY = args.confidence_strategy
    CONFIDENCE_ADAPTIVE = args.confidence_adaptive
    CONFIDENCE_MAX_THRESH = args.confidence_max_thresh

    OUT_DIR = args.out_dir

    CLIPPING_FACTOR = args.clipping_factor

    FEDPROX_FACTOR = args.fedprox_factor
    FEDPROX_ADAPTIVE = args.fedprox_adaptive

    DEBUG = args.debug

    USE_KMEANS = args.use_kmeans
    USE_ENTROPY = args.use_entropy
    USE_CLIPPING = args.use_clipping
    USE_FEDPROX = args.use_fedprox

    log(DEBUG, "Arguments read")

    TOTAL_MEM = TOTAL_MEM*(1024**3)

    # checking for dataset validity for selecting classes

    if (DATASET_NAME == "cifar10"):
        NUM_CLASSES = 10
    elif (DATASET_NAME == "cifar100"):
        NUM_CLASSES = 100
    elif (DATASET_NAME == 'pathmnist'):
        NUM_CLASSES = 9
    elif (DATASET_NAME == 'pneumoniamnist'):
        NUM_CLASSES = 2
    elif (DATASET_NAME == 'organamnist'):
        NUM_CLASSES = 11
    else:
        raise ValueError(f"{DATASET_NAME} has not been implemented yet!")

    # checking client distribuition validity for feddf hetero

    num_client_dict = np.sum(list(MODEL_LIST.values()))
    if (num_client_dict != NUM_CLIENTS):
        raise ValueError(
            f'Incompatible number of clients in model list w.r.t total number of clients')

    # seeding everything

    log(DEBUG, "Seeding RNG")

    if (SEED is not None):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

    if (CUDA_DETERMINISTIC):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    log(DEBUG, "RNG Seeded")

    # doing operations on output folder

    if (FED_STRATEGY == 'feddf' or FED_STRATEGY == 'feddf_hetero'):

        log(DEBUG, "Creating output folder for visualisation")

        experiment_time = f'{time.time_ns()}'
        experiment_dir = os.path.join(OUT_DIR, experiment_time)

        out_kmeans_folder = os.path.join(experiment_dir, 'kmean_visualisation')

        os.makedirs(out_kmeans_folder, exist_ok=True)

        log(DEBUG, "Folder created for visualisation")

    # setting up private datasets and test datasets

    log(DEBUG, "Commencing dataset download")

    train_data_path, test_set, test_labels = download_dataset(
        DATA_DIR, DATASET_NAME)
    kwargs_test_loader = {"num_workers": 1,
                          "pin_memory": False, "drop_last": False}
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, **kwargs_test_loader)

    log(DEBUG, "Dataset has been downloaded")

    # setting up distillation set if it is same as pvt set

    if (DISTILL_DATASET == DATASET_NAME):
        print(f"Same dataset for distillation and private training, splitting up test set into half...")
        split_test_loaders = split_standard(dataloader=test_loader, alpha=float(
            'inf'), batch_size=BATCH_SIZE, n_workers=1, seed=SEED)
        test_loader = split_test_loaders[0]
        distill_dataloader = split_test_loaders[1]

    log(DEBUG, "Doing fl partioning")

    fed_dir = do_fl_partitioning(
        train_data_path, NUM_CLIENTS, PARTITION_ALPHA, NUM_CLASSES, SEED, PARTITION_VAL_RATIO)

    log(DEBUG, "FL partitioning has been done")

    # creating local client functions

    if (FED_STRATEGY == "fedavg"):
        def client_fn(cid) -> client.fed_avg.FlowerClient:
            return client.fed_avg.FlowerClient(cid, MODEL_NAME, DATASET_NAME, fed_dir, BATCH_SIZE, CLIENT_CPUS, DEVICE, seed=SEED, cuda_deterministic=CUDA_DETERMINISTIC)

    elif (FED_STRATEGY == "feddf" or FED_STRATEGY == "feddf_hetero"):
        if (USE_CROPS):
            distill_dataloader = get_distill_imgloader(
                f'{DATA_DIR}/single_img_crops/crops', dataset_name=DATASET_NAME, batch_size=DISTILL_BATCH_SIZE, num_workers=1, distill_transforms=DISTILL_TRANSFORMS)
        else:
            if (DISTILL_DATASET != DATASET_NAME):
                distill_dataloader = create_std_distill_loader(
                    dataset_name=DISTILL_DATASET, transforms_name=DATASET_NAME, storage_path=DATA_DIR, n_images=NUM_TOTAL_IMAGES, batch_size=DISTILL_BATCH_SIZE, n_workers=1, seed=SEED, select_random=True, alpha=DISTILL_ALPHA, distill_transforms=DISTILL_TRANSFORMS)

        val_dataloader = combine_val_loaders(
            dataset_name=DATASET_NAME, path_to_data=fed_dir, n_clients=NUM_CLIENTS, batch_size=BATCH_SIZE, workers=1)

        model_init_list = []

        for model_name in MODEL_LIST.keys():
            for i in range(MODEL_LIST[model_name]):
                model_init_list.append(model_name)

        def client_fn(cid) -> client.fed_df.FlowerClient:
            if (FED_STRATEGY == "feddf"):
                return client.fed_df.FlowerClient(cid=cid, model_name=MODEL_NAME, dataset_name=DATASET_NAME, fed_dir=fed_dir, batch_size=BATCH_SIZE, num_cpu_workers=CLIENT_CPUS, device=DEVICE, debug=DEBUG, seed=SEED, cuda_deterministic=CUDA_DETERMINISTIC)
            else:
                return client.fed_df.FlowerClient(cid=cid, model_name=model_init_list[int(cid)], dataset_name=DATASET_NAME, fed_dir=fed_dir, batch_size=BATCH_SIZE, num_cpu_workers=CLIENT_CPUS, device=DEVICE, debug=DEBUG, seed=SEED, cuda_deterministic=CUDA_DETERMINISTIC)

    else:
        raise ValueError(f'{FED_STRATEGY} has not been implemented!')

    model_list = list(MODEL_LIST.keys())

    # starting central server with simulation start

    client_resources = {"num_cpus": CLIENT_CPUS}
    if (DEVICE.type == "cuda"):
        client_resources["num_gpus"] = CLIENT_GPUS

    ray_init_args = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
        "_memory": int(0.8*TOTAL_MEM),
        "num_cpus": TOTAL_CPUS,
        "num_gpus": TOTAL_GPUS,
        "object_store_memory": int(0.6*0.8*TOTAL_MEM)
    }

    if (FED_STRATEGY == "fedavg"):
        # FedAvg
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy.FedAvg(
                on_fit_config_fn=fed_avg_fn.get_fit_config_fn(
                    local_lr=LOCAL_LR, local_epochs=LOCAL_EPOCHS, adaptive_lr_round=USE_ADAPTIVE_LR_ROUND, adaptive_lr=USE_ADAPTIVE_LR, max_server_rounds=NUM_ROUNDS, use_fedprox=USE_FEDPROX, fedprox_factor=FEDPROX_FACTOR, use_clipping=USE_CLIPPING, clipping_factor=CLIPPING_FACTOR),
                on_evaluate_config_fn=fed_avg_fn.get_eval_config_fn(),
                initial_parameters=common_functions.initialise_parameters(
                    MODEL_NAME, DATASET_NAME),
                fit_metrics_aggregation_fn=common_functions.fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=common_functions.evaluate_metrics_aggregation_fn,
                evaluate_fn=fed_avg_fn.get_evaluate_fn(
                    MODEL_NAME, DATASET_NAME, test_loader, DEVICE),
                fraction_fit=FRACTION_FIT,
                fraction_evaluate=FRACTION_EVALUATE
            ),
            ray_init_args=ray_init_args,
            client_resources=client_resources,

        )

    elif (FED_STRATEGY == "feddf"):
        # FedDF
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=FedDF_strategy(
                num_rounds=NUM_ROUNDS,
                fraction_fit=FRACTION_FIT,
                fraction_evaluate=FRACTION_EVALUATE,
                distillation_dataloader=distill_dataloader,
                evaluation_dataloader=test_loader,
                evaluation_labels=test_labels,
                val_dataloader=val_dataloader,
                model_type=MODEL_NAME,
                dataset_name=DATASET_NAME,
                num_classes=NUM_CLASSES,
                device=DEVICE,
                initial_parameters=common_functions.initialise_parameters(
                    MODEL_NAME, DATASET_NAME),
                fit_metrics_aggregation_fn=common_functions.fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=common_functions.evaluate_metrics_aggregation_fn,
                on_fit_config_fn_client=fed_df_fn.get_on_fit_config_fn_client(
                    client_epochs=LOCAL_EPOCHS, client_lr=LOCAL_LR, clipping_factor=CLIPPING_FACTOR, use_clipping=USE_CLIPPING, use_adaptive_lr=USE_ADAPTIVE_LR, max_rounds=NUM_ROUNDS, use_adaptive_lr_round=USE_ADAPTIVE_LR_ROUND),
                on_fit_config_fn_server=fed_df_fn.get_on_fit_config_fn_server(
                    server_lr=SERVER_LR, distill_steps=SERVER_STEPS, use_early_stopping=USE_EARLY_STOPPING, early_stop_steps=SERVER_EARLY_STEPS, use_adaptive_lr=USE_ADAPTIVE_LR, warm_start=WARM_START, clipping_factor=CLIPPING_FACTOR, use_clipping=USE_CLIPPING, use_adaptive_steps=SERVER_STEPS_ADAPTIVE, adaptive_steps_min=SERVER_STEPS_ADAPTIVE_MIN, adaptive_steps_interval=SERVER_STEPS_ADAPTIVE_INTERVAL),
                evaluate_fn=fed_df_fn.evaluate_fn,
                warm_start_rounds=WARM_START_ROUNDS,
                debug=DEBUG,
                warm_start_interval=WARM_START_INTERVAL,
                kmeans_output_folder=out_kmeans_folder,
                num_total_images=NUM_TOTAL_IMAGES,
                kmeans_n_crops=NUM_DISTILL_IMAGES,
                kmeans_n_clusters=KMEANS_N_CLUSTERS,
                kmeans_heuristics=KMEANS_HEURISTICS,
                kmeans_mixed_factor=KMEANS_MIXED_FACTOR,
                kmeans_balancing=KMEANS_BALANCING,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                confidence_strategy=CONFIDENCE_STRATEGY,
                confidence_adaptive=CONFIDENCE_ADAPTIVE,
                confidence_max_thresh=CONFIDENCE_MAX_THRESH,
                fedprox_factor=FEDPROX_FACTOR,
                fedprox_adaptive=FEDPROX_ADAPTIVE,
                batch_size=DISTILL_BATCH_SIZE,
                num_cpu_workers=SERVER_CPUS,
                use_kmeans=USE_KMEANS,
                use_entropy=USE_ENTROPY,
                use_fedprox=USE_FEDPROX,
                seed=SEED,
                cuda_deterministic=CUDA_DETERMINISTIC,
            ),
            ray_init_args=ray_init_args,
            client_resources=client_resources,
        )

    elif (FED_STRATEGY == "feddf_hetero"):
        # FedDF Hetero
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=FedDF_hetero_strategy(
                num_rounds=NUM_ROUNDS,
                fraction_fit=FRACTION_FIT,
                fraction_evaluate=FRACTION_EVALUATE,
                distillation_dataloader=distill_dataloader,
                evaluation_dataloader=test_loader,
                evaluation_labels=test_labels,
                val_dataloader=val_dataloader,
                model_list=model_list,
                dataset_name=DATASET_NAME,
                num_classes=NUM_CLASSES,
                device=DEVICE,
                fit_metrics_aggregation_fn=common_functions.fit_metrics_aggregation_fn,
                evaluate_metrics_aggregation_fn=common_functions.evaluate_metrics_aggregation_fn,
                on_fit_config_fn_client=fed_df_fn.get_on_fit_config_fn_client(
                    client_epochs=LOCAL_EPOCHS, client_lr=LOCAL_LR, clipping_factor=CLIPPING_FACTOR, use_clipping=USE_CLIPPING, use_adaptive_lr=USE_ADAPTIVE_LR, max_rounds=NUM_ROUNDS, use_adaptive_lr_round=USE_ADAPTIVE_LR_ROUND),
                on_fit_config_fn_server=fed_df_fn.get_on_fit_config_fn_server(
                    server_lr=SERVER_LR, distill_steps=SERVER_STEPS, use_early_stopping=USE_EARLY_STOPPING, early_stop_steps=SERVER_EARLY_STEPS, use_adaptive_lr=USE_ADAPTIVE_LR, warm_start=WARM_START, clipping_factor=CLIPPING_FACTOR, use_clipping=USE_CLIPPING, use_adaptive_steps=SERVER_STEPS_ADAPTIVE, adaptive_steps_min=SERVER_STEPS_ADAPTIVE_MIN, adaptive_steps_interval=SERVER_STEPS_ADAPTIVE_INTERVAL),
                evaluate_fn=fed_df_fn.evaluate_fn,
                warm_start_rounds=WARM_START_ROUNDS,
                debug=DEBUG,
                warm_start_interval=WARM_START_INTERVAL,
                kmeans_output_folder=out_kmeans_folder,
                num_total_images=NUM_TOTAL_IMAGES,
                kmeans_n_crops=NUM_DISTILL_IMAGES,
                kmeans_n_clusters=KMEANS_N_CLUSTERS,
                kmeans_heuristics=KMEANS_HEURISTICS,
                kmeans_mixed_factor=KMEANS_MIXED_FACTOR,
                kmeans_balancing=KMEANS_BALANCING,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                confidence_strategy=CONFIDENCE_STRATEGY,
                confidence_adaptive=CONFIDENCE_ADAPTIVE,
                confidence_max_thresh=CONFIDENCE_MAX_THRESH,
                fedprox_factor=FEDPROX_FACTOR,
                fedprox_adaptive=FEDPROX_ADAPTIVE,
                batch_size=DISTILL_BATCH_SIZE,
                num_cpu_workers=SERVER_CPUS,
                use_kmeans=USE_KMEANS,
                use_entropy=USE_ENTROPY,
                use_fedprox=USE_FEDPROX,
                seed=SEED,
                cuda_deterministic=CUDA_DETERMINISTIC,
            ),
            ray_init_args=ray_init_args,
            client_resources=client_resources,
        )
