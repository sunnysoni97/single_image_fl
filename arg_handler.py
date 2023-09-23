import argparse


def parse_bool(inp_str: str) -> bool:
    if (inp_str == 'True' or inp_str == 'true'):
        return True
    elif (inp_str == 'False' or inp_str == 'false'):
        return False
    else:
        raise TypeError(
            f"{inp_str} is not a valid boolean value! Check passed arguments again.")


def parse_dict(inp_str: str) -> dict:
    res = eval(inp_str)
    return res


parser = argparse.ArgumentParser(
    description="FedAvg/FedDF Simulation using Flower")

parser.add_argument("--fed_strategy", type=str, default="feddf")
parser.add_argument("--model_name", type=str, default="resnet8")
parser.add_argument("--model_list", type=parse_dict,
                    default="{'resnet8':10,'resnet18':10}")

parser.add_argument("--num_clients", type=int, default=20)
parser.add_argument("--num_rounds", type=int, default=30)
parser.add_argument("--fraction_fit", type=float, default=0.4)
parser.add_argument("--fraction_evaluate", type=float, default=0.0)

parser.add_argument("--dataset_name", type=str, default="cifar10")
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--partition_alpha", type=float, default=100.0)
parser.add_argument("--partition_val_ratio", type=float, default=0.1)

parser.add_argument("--client_cpus", type=int, default=2)
parser.add_argument("--client_gpus", type=float, default=0.5)
parser.add_argument("--server_cpus", type=int, default=4)
parser.add_argument("--total_cpus", type=int, default=8)
parser.add_argument("--total_gpus", type=int, default=0)
parser.add_argument("--total_mem", type=int, default=8)

parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--local_epochs", type=int, default=40)
parser.add_argument("--local_lr", type=float, default=0.05)
parser.add_argument("--distill_batch_size", type=int, default=128)
parser.add_argument("--server_lr", type=float, default=0.005)
parser.add_argument("--server_steps", type=int, default=500)
parser.add_argument("--server_early_steps", type=int, default=5e2)
parser.add_argument("--use_early_stopping", type=parse_bool, default=False)
parser.add_argument("--use_adaptive_lr", type=parse_bool, default=False)
parser.add_argument("--use_adaptive_lr_round", type=parse_bool, default=False)

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--cuda_deterministic", type=parse_bool, default=False)

parser.add_argument("--use_crops", type=parse_bool, default=False)
parser.add_argument("--distill_dataset", type=str, default="cifar100")
parser.add_argument("--distill_alpha", type=float, default=1.0)
parser.add_argument("--num_distill_images", type=int, default=2250)
parser.add_argument("--num_total_images", type=int, default=100000)
parser.add_argument("--distill_transforms", type=str, default="v0")

parser.add_argument("--warm_start", type=parse_bool, default=True)
parser.add_argument("--warm_start_rounds", type=int, default=30)
parser.add_argument("--warm_start_interval", type=int, default=1)

parser.add_argument("--kmeans_n_clusters", type=int, default=10)
parser.add_argument("--kmeans_heuristics", type=str, default="mixed")
parser.add_argument("--kmeans_mixed_factor", type=str, default="50-50")
parser.add_argument("--kmeans_balancing", type=float, default=0.5)
parser.add_argument("--use_kmeans", type=parse_bool, default=True)

parser.add_argument("--confidence_threshold", type=float, default=0.1)
parser.add_argument("--confidence_strategy", type=str, default="top")
parser.add_argument("--confidence_adaptive", type=parse_bool, default=False)
parser.add_argument("--confidence_max_thresh", type=float, default=0.5)
parser.add_argument("--use_entropy", type=parse_bool, default=True)

parser.add_argument("--clipping_factor", type=float, default=1.0)
parser.add_argument("--use_clipping", type=parse_bool, default=True)

parser.add_argument("--fedprox_factor", type=float, default=1.0)
parser.add_argument("--fedprox_adaptive", type=parse_bool, default=False)
parser.add_argument("--use_fedprox", type=parse_bool, default=False)

parser.add_argument("--debug", type=parse_bool, default=False)
parser.add_argument("--out_dir", type=str, default="./out")
