from torch.utils.data import DataLoader
from typing import List
from data_loader_scripts.dl_common import create_lda_partitions
import torch
import numpy as np
from fed_df_data_loader.common import DistillDataset, get_distill_transforms
from torchvision.datasets import CIFAR100, CIFAR10
from pathlib import Path


def split_standard(dataloader: DataLoader, n_splits: int = 2, alpha: float = 1000, batch_size: int = 32, n_workers: int = 0, seed: int = None) -> List[DataLoader]:
    img_batches, label_batches = [], []
    for imgs, labels in dataloader:
        img_batches.append(imgs)
        label_batches.append(labels)

    all_imgs = torch.cat(img_batches)
    all_labels = torch.cat(label_batches).numpy()
    X = np.array(range(len(all_imgs)))
    dataset = [X, all_labels]

    partitions = create_lda_partitions(
        dataset, num_partitions=n_splits, concentration=alpha, accept_imbalanced=True, seed=seed)

    dataloader_list = []

    for i in range(n_splits):
        indices = partitions[0][i][0]
        labels = partitions[0][i][1]
        imgs = all_imgs[indices]
        new_dataset = DistillDataset(None, data=imgs, targets=labels)
        new_dataloader = DataLoader(
            new_dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True, shuffle=False)
        dataloader_list.append(new_dataloader)

    return dataloader_list


def create_std_distill_loader(dataset_name: str, storage_path: Path, n_images: int, transforms_name: str = "cifar10", alpha: float = 100.0, batch_size: int = 32, n_workers: int = 0, seed: int = None, distill_transforms: str = "v0") -> DataLoader:

    transform = get_distill_transforms(
        tgt_dataset=transforms_name, transform_type=distill_transforms)

    if(dataset_name == "cifar100"):
        full_dataset = CIFAR100(
            root=storage_path, train=True, transform=transform, download=True)
    elif(dataset_name == "cifar10"):
        full_dataset = CIFAR10(
            root=storage_path, train=True, transform=transform, download=True)
    else:
        raise ValueError("Dataset not implemented yet!")

    temp_dataloader = DataLoader(
        full_dataset, batch_size=1024, num_workers=n_workers, shuffle=False)

    Y = []
    for _, labels in temp_dataloader:
        Y.append(labels)

    Y = torch.cat(Y).numpy()

    total_len = len(full_dataset)
    n_partitions = total_len//n_images

    X = np.array(range(total_len))
    temp_dataset = [X, Y]
    partitions = create_lda_partitions(
        temp_dataset, num_partitions=n_partitions, concentration=alpha, accept_imbalanced=True, seed=seed)

    indices = partitions[0][0][0]
    all_labels = partitions[0][0][1]
    all_imgs = []
    for index in indices:
        all_imgs.append(full_dataset[index][0])
    new_dataset = DistillDataset(root=None, data=all_imgs, targets=all_labels)
    new_dataloader = DataLoader(new_dataset, batch_size=batch_size,
                                num_workers=n_workers, pin_memory=True, shuffle=False)

    return new_dataloader
