from torch.utils.data import DataLoader
from typing import List
from data_loader_scripts.dl_common import create_lda_partitions
import torch
import numpy as np
from fed_df_data_loader.common import DistillDataset


def split_standard(dataloader: DataLoader, n_splits: int = 2, alpha: float = 1000, batch_size: int = 32, n_workers: int = 0) -> List[DataLoader]:
    img_batches, label_batches = [], []
    for imgs, labels in dataloader:
        img_batches.append(imgs)
        label_batches.append(labels)

    all_imgs = torch.cat(img_batches)
    all_labels = torch.cat(label_batches).numpy()
    X = np.array(range(len(all_imgs)))
    dataset = [X, all_labels]

    partitions = create_lda_partitions(
        dataset, num_partitions=n_splits, concentration=alpha, accept_imbalanced=True)

    dataloader_list = []

    for i in range(n_splits):
        indices = partitions[0][i][0]
        labels = partitions[0][i][1]
        imgs = all_imgs[indices]
        new_dataset = DistillDataset(None, data=imgs, targets=labels)
        new_dataloader = DataLoader(
            new_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=False)
        dataloader_list.append(new_dataloader)

    return dataloader_list
