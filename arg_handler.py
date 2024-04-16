import argparse


def parse_bool(inp_str: str) -> bool:
    if inp_str == "True" or inp_str == "true":
        return True
    elif inp_str == "False" or inp_str == "false":
        return False
    else:
        raise TypeError(
            f"{inp_str} is not a valid boolean value! Check passed arguments again."
        )


def parse_dict(inp_str: str) -> dict:
    res = eval(inp_str)
    return res


parser = argparse.ArgumentParser(
    description="Single Image based fed learning simulation using Flower."
)

parser.add_argument(
    "--fed_strategy",
    type=str,
    default="feddf",
    help="Federated Strategy to use. Options :  fedavg, feddf, feddf_hetero",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="resnet8",
    help="Common NN architecture for clients and server (fedavg/feddf). Options : resnet[n], wresnet-[n]-[m]. Example : resnet8, wresnet-16-4",
)
parser.add_argument(
    "--model_list",
    type=parse_dict,
    default="{'resnet8':10,'resnet18':10}",
    help="NN architecture distribuition (feddf_hetero). Example : {'resnet8':5,'wresnet-16-4':5}",
)

parser.add_argument(
    "--num_clients", type=int, default=20, help="Total number of clients."
)
parser.add_argument(
    "--num_rounds", type=int, default=30, help="Total number of federated rounds."
)
parser.add_argument(
    "--fraction_fit",
    type=float,
    default=0.4,
    help="Ratio of clients to be selected for training every round.",
)
parser.add_argument(
    "--fraction_evaluate",
    type=float,
    default=0.0,
    help="Ratio of clients for client-evaluation every round.",
)

parser.add_argument(
    "--dataset_name",
    type=str,
    default="cifar10",
    help="Target dataset. Options : cifar10, cifar100, pathmnist, organamnist, pneumoniamnist",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="./data",
    help="Relative location of directory to download datasets.",
)
parser.add_argument(
    "--partition_alpha",
    type=float,
    default=100.0,
    help="Dirichtlet distribuition alpha for private dataset distribuition.",
)
parser.add_argument(
    "--partition_val_ratio",
    type=float,
    default=0.1,
    help="Ratio of validation examples from training set.",
)

parser.add_argument(
    "--client_cpus",
    type=int,
    default=2,
    help="Number of available cpus to each client.",
)
parser.add_argument(
    "--client_gpus",
    type=float,
    default=0.5,
    help="Number of available gpus to each client. Fraction acceptable.",
)
parser.add_argument(
    "--server_cpus",
    type=int,
    default=4,
    help="Number of cpus available to central server.",
)
parser.add_argument(
    "--total_cpus", type=int, default=8, help="Total number of cpus on the machine."
)
parser.add_argument(
    "--total_gpus", type=int, default=0, help="Total number of gpus on the machine."
)
parser.add_argument(
    "--total_mem",
    type=int,
    default=8,
    help="Total size of RAM on the machine (in gigabytes/ integer).",
)

parser.add_argument(
    "--batch_size", type=int, default=128, help="Batch size for local training."
)
parser.add_argument(
    "--local_epochs", type=int, default=40, help="Number of epochs for local training."
)
parser.add_argument(
    "--local_lr", type=float, default=0.05, help="LR for local training."
)
parser.add_argument(
    "--distill_batch_size",
    type=int,
    default=128,
    help="Batch size for global model training (server).",
)
parser.add_argument(
    "--server_lr", type=float, default=0.005, help="LR for global model training."
)
parser.add_argument(
    "--server_steps",
    type=int,
    default=500,
    help="Number of distillation steps (feddf).",
)
parser.add_argument(
    "--server_steps_adaptive",
    type=parse_bool,
    default=False,
    help="Enable/disable adaptive number of steps (feddf). Options : True/False",
)
parser.add_argument(
    "--server_steps_adaptive_min",
    type=int,
    default=50,
    help="Number of minimum distillation steps (adaptive).",
)
parser.add_argument(
    "--server_steps_adaptive_interval",
    type=int,
    default=5,
    help="Interval to change number of steps (adaptive).",
)
parser.add_argument(
    "--server_early_steps",
    type=int,
    default=5e2,
    help="Number of plateau steps for early stopping.",
)
parser.add_argument(
    "--use_early_stopping",
    type=parse_bool,
    default=False,
    help="Enable/disable early stopping (feddf). Options : True/False",
)
parser.add_argument(
    "--use_adaptive_lr",
    type=parse_bool,
    default=False,
    help="Enable/disable adaptive lr (local model training). Options : True/False",
)
parser.add_argument(
    "--use_adaptive_lr_round",
    type=parse_bool,
    default=False,
    help="Enable/disable adaptive lr (global model training). Options : True/False",
)

parser.add_argument(
    "--seed", type=int, default=None, help="Seed for RNG (for reproducible results)"
)
parser.add_argument(
    "--cuda_deterministic",
    type=parse_bool,
    default=False,
    help="Enable deterministic CUDA algorithms. Slow but deterministic.",
)

parser.add_argument(
    "--use_crops",
    type=parse_bool,
    default=False,
    help="Enable/disable use of single image crops (feddf). Options : True/False",
)
parser.add_argument(
    "--distill_dataset",
    type=str,
    default="cifar100",
    help="Dataset for distillation (if not using single image crops). Options same as dataset_name.",
)
parser.add_argument(
    "--distill_alpha",
    type=float,
    default=1.0,
    help="Dirichtlet dist. alpha for distillation dataset selection.",
)
parser.add_argument(
    "--num_distill_images",
    type=int,
    default=2250,
    help="Size of dataset used as distillation proxy (feddf/ integer).",
)
parser.add_argument(
    "--num_total_images",
    type=int,
    default=100000,
    help="Total number of images in distillation set.",
)
parser.add_argument(
    "--distill_transforms",
    type=str,
    default="v0",
    help="(experimental) changing transforms on image. Options: v0/v1",
)

parser.add_argument(
    "--warm_start",
    type=parse_bool,
    default=True,
    help="Enable/Disable warm start for FedDF. Options : True/False",
)
parser.add_argument(
    "--warm_start_rounds",
    type=int,
    default=30,
    help="Number of total rounds for warm start.",
)
parser.add_argument(
    "--warm_start_interval",
    type=int,
    default=1,
    help="Interval of rounds between warm starts.",
)

parser.add_argument(
    "--kmeans_n_clusters",
    type=int,
    default=10,
    help="Number of cluster (k) for KMeans selection.",
)
parser.add_argument(
    "--kmeans_heuristics",
    type=str,
    default="mixed",
    help="Heuristics for KMeans Selection. Options : mixed, easy, hard",
)
parser.add_argument(
    "--kmeans_mixed_factor",
    type=str,
    default="50-50",
    help="Ratio for mixed heuristic. Example : 50-50",
)
parser.add_argument(
    "--kmeans_balancing",
    type=float,
    default=0.5,
    help="Ratio for class balancing in KMeans selection. Example : 0.5",
)
parser.add_argument(
    "--use_kmeans",
    type=parse_bool,
    default=True,
    help="Enable/Disable KMeans selection for data pruning. Options : True/False",
)

parser.add_argument(
    "--confidence_threshold",
    type=float,
    default=0.1,
    help="Confidence Threshold for Entropy selection.",
)
parser.add_argument(
    "--confidence_strategy",
    type=str,
    default="top",
    help="Heuristics for Entropy Selection. Options : top, bottom, random",
)
parser.add_argument(
    "--confidence_adaptive",
    type=parse_bool,
    default=False,
    help="Disable/Enable adaptive threshold (entropy). Options : True/False",
)
parser.add_argument(
    "--confidence_max_thresh",
    type=float,
    default=0.5,
    help="Max threshold for adaptive pruning. Example : 0.5",
)
parser.add_argument(
    "--use_entropy",
    type=parse_bool,
    default=True,
    help="Enable/Disable entropy selection for data pruning. Options : True/False",
)

parser.add_argument(
    "--clipping_factor",
    type=float,
    default=1.0,
    help="Value for clipping factor (logit clipping)",
)
parser.add_argument(
    "--use_clipping",
    type=parse_bool,
    default=True,
    help="Enable/Disable use of logit clipping (for stability of learning)",
)

parser.add_argument(
    "--fedprox_factor",
    type=float,
    default=1.0,
    help="Factor value for fedprox strategy.",
)
parser.add_argument(
    "--fedprox_adaptive",
    type=parse_bool,
    default=False,
    help="Enable/Disable adaptive fedprox factor. Options : True/False",
)
parser.add_argument(
    "--use_fedprox",
    type=parse_bool,
    default=False,
    help="Enable/Disable use of fedprox term with FedAvg. Options : True/False",
)

parser.add_argument(
    "--debug",
    type=parse_bool,
    default=False,
    help="Enable/Disable debugging console messages. Options : True/False",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default="./out",
    help="Relative directory location for outputting results of the experiment.",
)
