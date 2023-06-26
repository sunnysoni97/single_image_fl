import torch
import pandas as pd
import numpy as np

import cupy
from cuml import KMeans as KMeans_GPU
from sklearn.cluster import KMeans as KMeans_CPU

from torch.utils.data import DataLoader, Dataset
from typing import Union, Tuple, List
from models import CifarResNet
from torchvision.models import ResNet

from torchvision.utils import make_grid, save_image
from io import BufferedWriter

from flwr.common import (
    ndarray_to_bytes,
    bytes_to_ndarray
)



# Function to form K-Mean clusters using the embeddings of the neural network


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

    df.drop(labels='embedding', axis=1, inplace=True)

    return df, c_model.inertia_


# Function to prune the clustered images using a specific heuristic
# supported heuristics : easy / hard / mixed
# heuristic percentage : easy-hard

def prune_clusters(raw_dataframe: pd.DataFrame, n_crops: int = 2250, heuristic: str = "mixed", heuristic_percentage: str = "50-50") -> pd.DataFrame:
    df = raw_dataframe.copy(True)
    df.insert(len(df.columns), "selected", "no")
    n_clusters = len(df['cluster'].value_counts().index)
    min_n_crops = int(n_crops/n_clusters * 50/100)

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

    df = df.groupby(by='cluster', group_keys=False).apply(select_minimum)

    out_df = df[df['selected'] == 'yes']

    df = df[df['selected'] == 'no']

    def fill_rest(x: pd.DataFrame):
        rem_crops = n_crops - int(min_n_crops*n_clusters)
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

    out_df = pd.concat([out_df, df]).reset_index(drop=True)
    out_df.index = list(out_df.index)
    out_df.index.name = 'img_no'

    out_df.drop('selected', axis=1, inplace=True)

    return out_df


# Function to visualise the images in the clusters

def visualise_clusters(cluster_df: pd.DataFrame, file: BufferedWriter, n_rows:int=10, n_cols:int=10) -> None:
    total_classes = n_rows*n_cols
    cluster_list = list(cluster_df.value_counts('cluster').index)
    
    if(total_classes > len(cluster_list)):
        total_classes = len(cluster_list)

    selected_clusters = [x for x in range(total_classes)]

    def get_images(df: pd.DataFrame, cluster_name: int, n_images: int = 1) -> List[np.ndarray]:
        imgs = df.groupby(by='cluster').get_group(cluster_name).sort_values(
            by='cluster_dist').loc[:, ["img", "cluster_dist"]].reset_index()
        easy_imgs = imgs.loc[0:n_images-1, "img"].tolist()
        return easy_imgs

    all_imgs = []
    for cluster in selected_clusters:
        cluster_imgs = get_images(cluster_df, cluster, 1)
        all_imgs += cluster_imgs

    all_imgs = [torch.tensor(x) for x in all_imgs]
    all_imgs = torch.cat(all_imgs, dim=0)

    grid_imgs = make_grid(tensor=all_imgs, nrow=n_rows,
                          pad_value=0.2, normalize=True)

    save_image(grid_imgs, file, format='png')
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
                 img_list:List[Union[torch.Tensor, np.ndarray]],
                 tgt_list:List[int]) -> None:
        super().__init__()
        self.data = img_list
        self.tgts = tgt_list

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img, tgt = self.data[index], self.tgts[index]
        if(not isinstance(img, torch.Tensor)):
            img = torch.tensor(img)

        return img,tgt
    
    def __len__(self) -> int:
        return len(self.data)


# function which returns dataloader from tranpsorted bytes

def extract_from_transport(img_bytes, batch_size:int=512, n_workers:int=2) -> DataLoader:
    img_np = bytes_to_ndarray(img_bytes)
    img_list = np.split(img_np, img_np.shape[0], axis=0)
    img_list = [x.squeeze() for x in img_list]
    
    dummy_targets = [1 for x in range(len(img_list))]

    new_dataset = KMeans_Dataset(img_list, dummy_targets)

    kwargs = {"batch_size": batch_size, "drop_last": False,
              "num_workers": n_workers, "pin_memory": False, "shuffle": False}
    distill_img_loader = DataLoader(new_dataset, **kwargs,)

    return distill_img_loader
    