from torch.utils.data import DataLoader
from typing import List
import torch
import numpy as np
from torchvision.datasets import CIFAR100, CIFAR10
from pathlib import Path

from data_loader_scripts.dl_common import create_lda_partitions
from data_loader_scripts.create_dataloader import TorchVision_FL
from fed_df_data_loader.common import DistillDataset, get_distill_transforms


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
            new_dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=False, shuffle=False)
        dataloader_list.append(new_dataloader)

    return dataloader_list


def create_std_distill_loader(dataset_name: str, storage_path: Path, n_images: int, transforms_name: str = "cifar10", select_random: bool = True, alpha: float = 100.0, batch_size: int = 32, n_workers: int = 0, seed: int = None, distill_transforms: str = "v0") -> DataLoader:

    transform = get_distill_transforms(
        tgt_dataset=transforms_name, transform_type=distill_transforms)

    if (dataset_name == "cifar100"):
        full_dataset = CIFAR100(
            root=storage_path, train=True, download=True)
    elif (dataset_name == "cifar10"):
        full_dataset = CIFAR10(
            root=storage_path, train=True, download=True)
    else:
        raise ValueError("Dataset not implemented yet!")

    total_len = len(full_dataset)
    X = np.array(range(total_len))

    if (select_random):
        indices = np.random.choice(a=X, size=n_images, replace=False)
        all_labels = []
        for idx in indices:
            all_labels.append(full_dataset[idx][1])
        all_labels = np.array(all_labels, dtype=np.int8)

    else:
        temp_dataloader = DataLoader(
            full_dataset, batch_size=1024, num_workers=n_workers, shuffle=False)

        Y = []
        for _, labels in temp_dataloader:
            Y.append(labels)
        Y = torch.cat(Y).numpy()

        n_partitions = total_len//n_images
        temp_dataset = [X, Y]

        partitions = create_lda_partitions(
            temp_dataset, num_partitions=n_partitions, concentration=alpha, accept_imbalanced=True, seed=seed)

        indices = partitions[0][0][0]
        all_labels = partitions[0][0][1]

    all_imgs = []
    for index in indices:
        all_imgs.append(full_dataset[index][0])
    new_dataset = TorchVision_FL(
        data=all_imgs, targets=all_labels, transform=transform)
    kwargs = {"batch_size": batch_size, "drop_last": False,
              "num_workers": n_workers, "pin_memory": True, "shuffle": False}
    new_dataloader = DataLoader(new_dataset, **kwargs)
    # new_dataset = DistillDataset(root=None, data=all_imgs, targets=all_labels)
    # new_dataloader = DataLoader(new_dataset, batch_size=batch_size,
    #                             num_workers=n_workers, pin_memory=False, shuffle=False)

    return new_dataloader
