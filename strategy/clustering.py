import torch
import pandas as pd
import numpy as np

import cupy
from cuml import KMeans as KMeans_GPU
from sklearn.cluster import KMeans as KMeans_CPU

from torch.utils.data import DataLoader
from typing import Union, Tuple
from models import CifarResNet
from torchvision.models import ResNet


def cluster_embeddings(dataloader: DataLoader, model: Union[CifarResNet, ResNet], device: torch.device, n_clusters: int = 10, seed: int = 1) -> Tuple[pd.DataFrame, float]:

    # creating dataframe
    df = pd.DataFrame(columns=['img', 'embedding'])

    # performing forward inference for getting embeddings
    model.to(device)
    model.eval()

    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            output = model.forward_avgpool(imgs)
            imgs_list = np.split(imgs.cpu().numpy(), imgs.size()[0], axis=0)
            embedding_list = np.split(
                output.cpu().numpy(), output.size()[0], axis=0)
            new_data = pd.DataFrame(
                data={'img': imgs_list, 'embedding': embedding_list})
            df = pd.concat([df, new_data]).reset_index(drop=True)

    # initialising cluster model

    c_model = KMeans_GPU(n_clusters=n_clusters, n_init=1, random_state=seed) if device == torch.device(
        'cuda') else KMeans_CPU(n_clusters=n_clusters, n_init=1, random_state=seed)

    # fitting clusters, calculating distance and cluster labels

    all_embedding = np.concatenate(df['embedding'].to_list())
    if (device == torch.device('cuda')):
        all_embedding = torch.tensor(data=all_embedding, device=device)

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

    return df, c_model.inertia_
