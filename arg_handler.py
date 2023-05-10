import argparse

parser = argparse.ArgumentParser(description="FedAvg Simulation using Flower")

parser.add_argument("--fed_strategy", type=str, default="fedavg")

parser.add_argument("--num_clients", type=int, default=20)
parser.add_argument("--num_rounds", type=int, default=10)
parser.add_argument("--fraction_fit", type=float, default=0.4)
parser.add_argument("--fraction_evaluate", type=float, default=0.0)

parser.add_argument("--dataset_name", type=str, default="cifar10")
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--partition_alpha", type=float, default=100.0)
parser.add_argument("--partition_val_ratio", type=float, default=0.1)

parser.add_argument("--client_cpus", type=int, default=2)
parser.add_argument("--client_gpus", type=float, default=0.5)
parser.add_argument("--server_cpus", type=int, default=4)

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--local_epochs", type=int, default=20)
parser.add_argument("--local_lr", type=float, default=0.005)
parser.add_argument("--server_lr", type=float, default=1e-3)
parser.add_argument("--server_steps", type=int, default=1e3)
parser.add_argument("--server_early_steps", type=int, default=5e2)
parser.add_argument("--use_early_stopping", type=bool, default=True)
parser.add_argument("--use_adaptive_lr", type=bool, default=True)

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--cuda_deterministic", type=bool, default=False)

parser.add_argument("--use_crops", type=bool, default=False)
parser.add_argument("--distill_dataset", type=str, default="cifar100")
parser.add_argument("--num_distill_images", type=int, default=1000)
