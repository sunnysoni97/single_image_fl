from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split
from typing import Tuple
import torch


# FUNCTION FOR RETURNING DATALOADERS TO CLIENTS

def load_dataset(dataset_name: str, n_clients: int, batch_size: int, seed_value: int) -> Tuple[DataLoader, DataLoader, DataLoader]:

    # creating preprocessor transformation
    if(dataset_name == "cifar10"):
        normalize = transforms.Normalize(
            (0.49, 0.48, 0.45), (0.25, 0.24, 0.26))
    elif(dataset_name == "cifar100"):
        normalize = transforms.Normalize(
            (0.51, 0.49, 0.44), (0.27, 0.26, 0.28))
    else:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
    transform = transforms.Compose([
        transforms.ToTensor(), normalize
    ])

    # loading the dataset
    if(dataset_name == "cifar10"):
        train_set = CIFAR10("./dataset", train=True,
                            download=True, transform=transform)
        test_set = CIFAR10("./dataset", train=False,
                           download=True, transform=transform)
    elif(dataset_name == "cifar100"):
        train_set = CIFAR100("./dataset", train=True,
                             download=True, transform=transform)
        test_set = CIFAR100("./dataset", train=False,
                            download=True, transform=transform)
    else:
        raise NotImplementedError(
            f"Specified dataset : {dataset_name} not available.")

    # stop if number of clients are negative
    if(n_clients < 0):
        raise ValueError(f'{n_clients} is not valid for number of clients!')

    # generating private datasets
    partition_size = len(train_set) // n_clients
    lengths = [partition_size] * n_clients
    datasets = random_split(
        train_set, lengths, torch.Generator().manual_seed(seed_value))

    # generating private loaders
    if(batch_size < 0):
        raise ValueError(f'{batch_size} is not valid for batch size!')
    train_loaders = []
    val_loaders = []
    for ds in datasets:
        len_val = len(ds) // 10
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(
            ds, lengths, torch.Generator().manual_seed(seed_value))
        train_loaders.append(DataLoader(
            dataset=ds_train, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(
            dataset=ds_val, batch_size=batch_size, shuffle=False))

    # generating common test loader
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False)

    return (train_loaders, val_loaders, test_loader)
