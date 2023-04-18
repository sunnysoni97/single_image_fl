from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import torch
import numpy as np
from pathlib import Path
import os


# Transforms for CIFAR 10 test set
def cifar10_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ]
    )

# Transforms for CIFAR 100 test set


def cifar100_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ]
    )

# Downloads complete training dataset on the disk, returns path of combined training data file and test set in the memory


def download_dataset(data_storage_path="./data", dataset_name="cifar10"):

    if(dataset_name == "cifar10"):
        train_set = CIFAR10(root=data_storage_path, train=True, download=True)
        test_set = CIFAR10(root=data_storage_path, train=False,
                           transform=cifar10_transforms())

    elif(dataset_name == "cifar100"):
        train_set = CIFAR100(root=data_storage_path, train=True, download=True)
        test_set = CIFAR100(root=data_storage_path,
                            train=False, transform=cifar100_transforms())
    else:
        raise ValueError("This dataset is not implemented yet!")

    train_data_path = Path(data_storage_path) / \
        f"{dataset_name}-federated"
    if(not os.path.exists(train_data_path)):
        os.makedirs(train_data_path)
    train_file_path = train_data_path / "training.pt"
    print(f"Generating unified {dataset_name} dataset")
    torch.save([train_set.data, np.array(train_set.targets)], train_file_path)

    return train_file_path, test_set
