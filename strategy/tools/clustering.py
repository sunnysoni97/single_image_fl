import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

import cupy
from cuml.common.device_selection import using_device_type
from cuml import KMeans as KMeans_GPU
from sklearn.cluster import KMeans as KMeans_CPU

from cuml import TSNE as TSNE_GPU
from sklearn.manifold import TSNE as TSNE_CPU
from cuml.neighbors import NearestNeighbors

from torch.utils.data import DataLoader, Dataset
from typing import Union, Tuple, List
from models import CifarResNet
from torchvision.models import ResNet

from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from io import BufferedWriter

from flwr.common import (
    ndarray_to_bytes,
    bytes_to_ndarray
)


# Function to form K-Mean clusters using the embeddings of the neural network


def cluster_embeddings(dataloader: DataLoader, model: Union[CifarResNet, ResNet], device: torch.device, n_clusters: int = 10, seed: int = 1) -> Tuple[pd.DataFrame, float]:

    # creating dataframe
    df = pd.DataFrame(columns=['img', 'embedding', 'pred'])

    # performing forward inference for getting embeddings
    model.to(device)
    model.eval()

    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            output = model.forward_avgpool(imgs)
            preds_softmax = F.softmax(input=model(imgs), dim=-1)
            preds = torch.argmax(input=preds_softmax,
                                 dim=-1).cpu().numpy().tolist()
            confs = torch.amax(input=preds_softmax, dim=-
                               1).cpu().numpy().tolist()
            imgs = imgs.cpu().numpy()
            output = output.cpu().numpy()
            imgs_list = np.split(imgs, imgs.shape[0], axis=0)
            embedding_list = np.split(
                output, output.shape[0], axis=0)
            new_data = pd.DataFrame(
                data={'img': imgs_list, 'embedding': embedding_list, 'pred': preds, 'conf': confs})
            df = pd.concat([df, new_data], ignore_index=True)

    # initialising cluster model

    c_model = KMeans_GPU(n_clusters=n_clusters, n_init=1, random_state=seed) if device == torch.device(
        'cuda') else KMeans_CPU(n_clusters=n_clusters, n_init=1, random_state=seed)

    # fitting clusters, calculating distance and cluster labels

    all_embedding = np.concatenate(df['embedding'].to_list())
    if (device == torch.device('cuda')):
        all_embedding = torch.tensor(data=all_embedding).to(device)

    l2dist = c_model.fit_transform(all_embedding)

    # selecting distance to closest cluster and putting in df

    clusters = cupy.asnumpy(c_model.labels_) if device == torch.device(
        'cuda') else c_model.labels_
    assigned_clusters = np.split(
        clusters, indices_or_sections=clusters.shape[0], axis=0)
    for i in range(len(assigned_clusters)):
        assigned_clusters[i] = assigned_clusters[i].item()

    assigned_distance = []

    i = 0
    for cluster in assigned_clusters:
        assigned_distance.append(l2dist[i][cluster].item())
        i += 1

    df.insert(len(df.columns), 'cluster', assigned_clusters)
    df.insert(len(df.columns), 'cluster_dist', assigned_distance)

    df.index = list(df.index)
    df.index.name = 'img_no'

    return df, c_model.inertia_


# Function to prune the clustered images using a specific heuristic
# supported heuristics : easy / hard / mixed
# heuristic percentage : easy-hard

def prune_clusters(raw_dataframe: pd.DataFrame, n_crops: int = 2250, heuristic: str = "mixed", heuristic_percentage: str = "50-50", kmeans_balancing: float = 0.5) -> pd.DataFrame:
    df = raw_dataframe.copy(True)
    df.insert(len(df.columns), "selected", "no")
    n_classes = len(df['pred'].value_counts().index)
    balance_percent = kmeans_balancing
    min_n_crops = int(n_crops/n_classes * balance_percent)

    if (heuristic == 'easy'):
        percent_easy = 100
        percent_hard = 50

    elif (heuristic == 'hard'):
        percent_easy = 0
        percent_hard = 100

    elif (heuristic == 'mixed'):
        percent_list = heuristic_percentage.split(sep='-')
        percent_list = [int(x) for x in percent_list]

        if (not (len(percent_list) == 2 and type(percent_list[0]) == type(1) and percent_list[0]+percent_list[1] == 100)):
            raise ValueError(
                f'{heuristic_percentage} is not correct way to pass percentage!')

        percent_easy = percent_list[0]
        percent_hard = percent_list[1]

    else:
        raise NotImplementedError(
            f'{heuristic} has not been implemented for our kmeans!')

    def select_minimum(x: pd.DataFrame):
        n_easy = int(percent_easy/100*min_n_crops)
        n_hard = int(percent_hard/100*min_n_crops)

        if (n_easy + n_hard != min_n_crops):
            n_easy += (min_n_crops-(n_easy+n_hard))

        x = x.sort_values(by='cluster_dist', ascending=True)
        x.iloc[0:n_easy, x.columns.get_loc('selected')] = 'yes'

        x = x.sort_values(by='cluster_dist', ascending=False)
        x.iloc[0:n_hard, x.columns.get_loc('selected')] = 'yes'

        x = x.sort_values(by='img_no', ascending=True)

        return x

    df = df.groupby(by='pred', group_keys=False).apply(select_minimum)

    out_df = df[df['selected'] == 'yes']

    df = df[df['selected'] == 'no']

    def fill_rest(x: pd.DataFrame):
        rem_crops = n_crops - len(out_df)
        n_easy = int(percent_easy/100*rem_crops)
        n_hard = int(percent_hard/100*rem_crops)

        if (n_easy + n_hard != rem_crops):
            n_easy += (rem_crops-(n_easy+n_hard))

        x = x.sort_values(by='cluster_dist', ascending=True)
        x.iloc[0:n_easy, x.columns.get_loc('selected')] = 'yes'

        x = x.sort_values(by='cluster_dist', ascending=False)
        x.iloc[0:n_hard, x.columns.get_loc('selected')] = 'yes'

        x = x.sort_values(by='img_no', ascending=True)

        return x

    df = fill_rest(df)
    df = df[df['selected'] == 'yes']

    out_df = pd.concat([out_df, df], ignore_index=True)
    out_df.index = list(out_df.index)
    out_df.index.name = 'img_no'

    out_df.drop('selected', axis=1, inplace=True)

    return out_df


# Functions to calculate and visualise tSNE distribution and clusters based on tsne

# TSNE calculation function


def calculate_tsne(cluster_df: pd.DataFrame, device: torch.device, n_cpu: int = -1) -> pd.DataFrame:
    img_embeddings = np.array(cluster_df["embedding"].to_list()).squeeze()

    if (device == torch.device('cuda')):
        img_embeddings = cupy.asnumpy(img_embeddings)

    t_model = TSNE_GPU(n_components=2, method='barnes_hut', output_type='numpy') if device == torch.device(
        'cuda') else TSNE_CPU(n_components=2, method='barnes_hut', n_jobs=n_cpu)
    tsne_embeddings = t_model.fit_transform(img_embeddings)

    out_df = cluster_df.copy()
    tsne_embeddings = np.split(tsne_embeddings, len(tsne_embeddings), axis=0)

    out_df.insert(loc=len(out_df.columns),
                  column="tsne_embeddings", value=tsne_embeddings)
    return out_df

# TSNE visualisation function


def _scale_tsne(tsne_vals: np.ndarray) -> np.ndarray:
    tsne_values = tsne_vals.copy()
    min_val = np.min(tsne_values, axis=0, keepdims=True)
    max_val = np.max(tsne_values, axis=0, keepdims=True)
    min_val = np.repeat(a=min_val, repeats=tsne_values.shape[0], axis=0)
    max_val = np.repeat(a=max_val, repeats=tsne_values.shape[0], axis=0)
    range_val = max_val-min_val
    tsne_values = tsne_values-min_val
    tsne_values = np.divide(tsne_values, range_val)
    return tsne_values

# for visualising clusters in a grid (manifold)


def visualise_clusters(cluster_df: pd.DataFrame, file: BufferedWriter, device: torch.device = torch.device('cpu'), grid_size: int = 10) -> None:

    x_coords = np.linspace(0, 1.0, grid_size+1)
    x_coords = (x_coords[:-1] + np.roll(x_coords, -1)[:-1])/2
    y_coords = np.linspace(0, 1.0, grid_size+1)
    y_coords = (y_coords[:-1] + np.roll(y_coords, -1)[:-1])/2

    all_coords = np.array(np.meshgrid(x_coords, y_coords,
                          indexing='ij')).T.reshape(-1, 2)

    neighbour_model = NearestNeighbors(n_neighbors=1, output_type='numpy')

    device_str = "GPU" if device == torch.device('cuda') else "CPU"

    tsne_values = np.array(
        cluster_df['tsne_embeddings'].to_list()).squeeze(axis=1)
    tsne_values = _scale_tsne(tsne_vals=tsne_values)

    with using_device_type(device_str):
        neighbour_model.fit(tsne_values)
        _, img_idxs = neighbour_model.kneighbors(all_coords)

    idxs_shape = img_idxs.shape
    img_idxs = np.flip(m=img_idxs.reshape(-1, grid_size),
                       axis=0).reshape(idxs_shape)

    imgs_np = np.array(cluster_df['img'].to_list()).squeeze(axis=1)

    selected_imgs = imgs_np[img_idxs]
    selected_imgs = torch.tensor(selected_imgs).squeeze(dim=1)

    grid_imgs = make_grid(tensor=selected_imgs,
                          nrow=grid_size, pad_value=0.2, normalize=True)

    save_image(grid_imgs, file, format='png')
    return


# for visualising scatter plot of all clusters

def visualise_tsne(tsne_df: pd.DataFrame, out_file: BufferedWriter, n_classes: int, label_metadata: list = None, round_no: int = 0) -> None:

    tsne_values = np.array(tsne_df['tsne_embeddings'].to_list()).squeeze()

    # scaling values between 0 and 1
    tsne_values = _scale_tsne(tsne_vals=tsne_values)

    # creating a new df for plotting tsne data according to cluster
    new_df = pd.DataFrame(data={'pred': tsne_df['pred'], 'tsne_values': np.split(
        tsne_values, tsne_values.shape[0])})
    x_vals = []
    y_vals = []
    labels = []

    # collecting x and y data by clusters
    for group in new_df['pred'].value_counts().index:
        grp_df = new_df.groupby(by='pred').get_group(group)
        tsne_values = np.array(grp_df['tsne_values'].to_list()).squeeze(axis=1)
        x_vals_grp = tsne_values[:, 0]
        y_vals_grp = tsne_values[:, 1]
        x_vals.append(x_vals_grp)
        y_vals.append(y_vals_grp)
        if (label_metadata):
            labels.append(label_metadata[group])
        else:
            labels.append(group)

    # plotting time
    colors = np.linspace(0.0, 1.0, n_classes)
    cmap = plt.get_cmap('rainbow')

    for i in range(len(labels)):
        c_idx = int(labels[i]) if not label_metadata else label_metadata.index(
            labels[i])
        plt.scatter(x=x_vals[i], y=y_vals[i], s=10, c=[cmap(colors[c_idx]) for _ in x_vals[i]], label=str(
            labels[i]), alpha=0.8, linewidths=0, edgecolors='none')

    bbox = (1.05, 0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = zip(
        *sorted(zip(handles, labels), key=lambda t: int(t[1]) if not label_metadata else t[1]))
    ncols = np.ceil((len(labels)/15))
    plt.legend(handles=handles, labels=labels, loc='center left',
               bbox_to_anchor=bbox, ncols=ncols, title='Predicted Label')
    plt.title(f'tSNE (n=2) at Round {round_no}')
    plt.savefig(fname=out_file, format='png', bbox_inches='tight')
    plt.close()

    return

# Function to prepare for images for transport


def prepare_for_transport(pruned_df: pd.DataFrame):
    distill_imgs = np.array(pruned_df.loc[:, "img"].tolist()).squeeze()
    distill_imgs = ndarray_to_bytes(distill_imgs)
    return distill_imgs

# Function to extract dataloader from the transported images

# Class for supporting creation of dataloader


class KMeans_Dataset(Dataset):
    def __init__(self,
                 img_list: List[Union[torch.Tensor, np.ndarray]],
                 tgt_list: List[int]) -> None:
        super().__init__()
        self.data = img_list
        self.tgts = tgt_list

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img, tgt = self.data[index], self.tgts[index]
        if (not isinstance(img, torch.Tensor)):
            img = torch.tensor(img)

        return img, tgt

    def __len__(self) -> int:
        return len(self.data)


# function which returns dataloader from tranpsorted bytes

def extract_from_transport(img_bytes, batch_size: int = 512, n_workers: int = 2) -> DataLoader:
    img_np = bytes_to_ndarray(img_bytes)
    img_list = np.split(img_np, img_np.shape[0], axis=0)

    if (len(img_list[0].shape) == 4):
        img_list = [x.squeeze() for x in img_list]

    dummy_targets = [1 for x in range(len(img_list))]

    new_dataset = KMeans_Dataset(img_list, dummy_targets)

    kwargs = {"batch_size": batch_size, "drop_last": False,
              "num_workers": n_workers, "pin_memory": False, "shuffle": False}
    distill_img_loader = DataLoader(new_dataset, **kwargs,)

    return distill_img_loader
