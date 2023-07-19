from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import torch
import numpy as np
from pathlib import Path
import os
from medmnist.info import INFO
import medmnist

# Transforms for CIFAR 10 test set


def cifar10_transforms(is_train: bool = True):
    t_compose = transforms.Compose([])

    if (is_train):
        t_compose.transforms.extend([transforms.RandomHorizontalFlip(),
                                     transforms.RandomCrop((32, 32), 4),
                                     ])

    t_compose.transforms.extend([transforms.ToTensor(),
                                 transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                      (0.2023, 0.1994, 0.2010))])

    return t_compose

# Transforms for CIFAR 100 test set


def cifar100_transforms(is_train: bool = True):

    t_compose = transforms.Compose([])

    if (is_train):
        t_compose.transforms.extend([transforms.RandomHorizontalFlip(),
                                     transforms.RandomCrop((32, 32), 4),
                                     ])

    t_compose.transforms.extend([transforms.ToTensor(),
                                 transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                      (0.2675, 0.2565, 0.2761))])

    return t_compose


# Transforms for MedMNIST

def medmnist_transforms(is_train: bool = True):
    t_compose = transforms.Compose([])

    if (is_train):
        t_compose.transforms.extend([transforms.RandomCrop((28, 28), 4)])

    t_compose.transforms.extend([transforms.ToTensor(),
                                 transforms.Normalize([0.5], [0.5])])

    return t_compose

# Returns appropriate transforms


def get_transforms(dataset_name: str = "cifar10", is_train: bool = True):
    if (dataset_name == "cifar10"):
        transformF = cifar10_transforms(is_train=is_train)
    elif (dataset_name == "cifar100"):
        transformF = cifar100_transforms(is_train=is_train)
    elif (dataset_name in ['pathmnist', 'pneumoniamnist', 'organamnist']):
        transformF = medmnist_transforms(is_train=is_train)
    else:
        raise ValueError(f'{dataset_name} not implemented yet!')
    return transformF

# Downloads complete training dataset on the disk, returns path of combined training data file and test set in the memory


def download_dataset(data_storage_path="./data", dataset_name="cifar10"):

    if (dataset_name == "cifar10"):
        train_set = CIFAR10(root=data_storage_path, train=True, download=True)
        test_set = CIFAR10(root=data_storage_path, train=False,
                           transform=get_transforms(dataset_name, is_train=False))

    elif (dataset_name == "cifar100"):
        train_set = CIFAR100(root=data_storage_path, train=True, download=True)
        test_set = CIFAR100(root=data_storage_path,
                            train=False, transform=get_transforms(dataset_name, is_train=False))
    elif (dataset_name in ['pathmnist', 'pneumoniamnist', 'organamnist']):
        py_class_name = INFO[dataset_name]['python_class']
        py_class = getattr(medmnist, py_class_name)
        train_set = py_class(root=data_storage_path,
                             download=True, split='train')
        test_set = py_class(root=data_storage_path, transform=get_transforms(
            dataset_name, is_train=False), split='test')

    else:
        raise ValueError("This dataset is not implemented yet!")

    train_data_path = Path(data_storage_path) / \
        f"{dataset_name}-federated"
    if (not os.path.exists(train_data_path)):
        os.makedirs(train_data_path)
    train_file_path = train_data_path / "training.pt"
    print(f"Generating unified {dataset_name} dataset")
    if (dataset_name in ['cifar10', 'cifar100']):
        torch.save([train_set.data, np.array(
            train_set.targets)], train_file_path)
    else:
        torch.save([train_set.imgs, np.array(
            train_set.labels)], train_file_path)

    return train_file_path, test_set
