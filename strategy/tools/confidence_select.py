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


def prune_confident_crops(model: Union[CifarResNet, ResNet], device: torch.device, cluster_df: pd.DataFrame, confidence_threshold: float = 0.1, batch_size: int = 128, num_workers: int = 0):

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
            # pred_labels = torch.argmax(input=preds, dim=-1)
            conf_list.append(confs)
            # pred_list.append(pred_labels)

    all_conf = torch.cat(conf_list, dim=0).cpu().numpy()

    # creating a new dataframe for selection
    new_df = cluster_df.copy()
    selected = ['yes' for i in range(len(new_df))]
    new_df['conf_value'] = all_conf
    new_df['selected'] = selected

    # removing top k percentile of crops per class

    def remove_topk(x: pd.DataFrame):
        n = int(confidence_threshold*len(x))

        x = x.sort_values(by='conf_value', ascending=False)
        x.iloc[0:n, x.columns.get_loc('selected')] = 'no'

        return x

    new_df = new_df.groupby(by='pred', group_keys=False).apply(remove_topk)

    new_df = new_df[new_df['selected'] == 'yes']
    new_df.drop('selected', axis=1, inplace=True)
    new_df.sort_index(inplace=True)

    return new_df
