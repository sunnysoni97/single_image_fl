import pandas as pd
import numpy as np
from models import CifarResNet, ResNet
from typing import Union

import torch
from torch.utils.data import DataLoader
from fed_df_data_loader.common import DistillDataset

import torch.nn.functional as F


def __create_dl(cluster_df: pd.DataFrame, batch_size: int = 128, num_workers: int = 0) -> DataLoader:

    img_list = torch.Tensor(
        np.array(cluster_df["img"].to_list()).squeeze(axis=1))
    target_list = torch.zeros(size=(img_list.size()[0], 1))

    dataset = DistillDataset(root=None, data=img_list, targets=target_list)
    dl = DataLoader(dataset=dataset, pin_memory=False, shuffle=False,
                    batch_size=batch_size, num_workers=num_workers)

    return dl


def prune_confident_crops(model: Union[CifarResNet, ResNet], device: torch.device, cluster_df: pd.DataFrame, confidence_threshold: float = 0.5, min_crops: int = 2000, batch_size: int = 128, num_workers: int = 0):

    # creating dataloader for inference
    dl = __create_dl(cluster_df=cluster_df,
                     batch_size=batch_size, num_workers=num_workers)

    # inferencing to get confidence level
    model.to(device)
    conf_list = []

    with torch.no_grad():
        for imgs, _ in dl:
            imgs = imgs.to(device)
            preds = model(imgs)
            preds = F.softmax(preds, dim=-1)
            confs = torch.amax(preds, dim=-1)
            conf_list.append(confs)

    all_conf = torch.cat(conf_list, dim=0).cpu().numpy()

    # creating a new dataframe for selection
    new_df = cluster_df.copy()
    new_df.insert(loc=len(new_df.columns), column='conf_value', value=all_conf)

    # selecting on the basis of confidence threshold
    select_df = new_df[new_df['conf_value'] < confidence_threshold]

    # checking if it meets minimum crops criteria, else using sorting for selection
    if (len(select_df) < min_crops):
        new_df.sort_values(by='conf_value', ignore_index=True, inplace=True)
        select_df = new_df.iloc[0:min_crops]

    return select_df
