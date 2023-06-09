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

    df.index = list(df.index)
    df.index.name = 'img_no'

    df.drop(labels='embedding', axis=1, inplace=True)

    return df, c_model.inertia_


# Function to prune the clustered images using a specific heuristic
# supported heuristics : easy / hard / mixed
# heuristic percentage : easy-hard

def prune_clusters(raw_dataframe:pd.DataFrame, n_crops:int=2250, heuristic:str="mixed", heuristic_percentage:str="50-50") -> pd.DataFrame:
    df = raw_dataframe.copy(True)
    df.insert(len(df.columns),"selected","no")
    n_clusters = len(df['cluster'].value_counts().index)
    min_n_crops = int(n_crops/n_clusters * 50/100)
    
    if(heuristic=='easy'):
        percent_easy = 100
        percent_hard = 50

    elif(heuristic=='hard'):
        percent_easy = 0
        percent_hard = 100

    elif(heuristic=='mixed'):
        percent_list = heuristic_percentage.split(sep='-')
        percent_list = [int(x) for x in percent_list]
        
        if(not (len(percent_list)==2 and type(percent_list[0])==type(1) and percent_list[0]+percent_list[1] == 100)):
            raise ValueError(f'{heuristic_percentage} is not correct way to pass percentage!')
        
        percent_easy = percent_list[0]
        percent_hard = percent_list[1]

    else:
        raise NotImplementedError(f'{heuristic} has not been implemented for our kmeans!')

    def select_minimum(x:pd.DataFrame):
        n_easy = int(percent_easy/100*min_n_crops)
        n_hard = int(percent_hard/100*min_n_crops)
        
        if(n_easy + n_hard != min_n_crops):
            n_easy += (min_n_crops-(n_easy+n_hard))

        x = x.sort_values(by='cluster_dist',ascending=True)
        x.iloc[0:n_easy, x.columns.get_loc('selected')] = 'yes'

        x = x.sort_values(by='cluster_dist',ascending=False)
        x.iloc[0:n_hard, x.columns.get_loc('selected')] = 'yes'

        x = x.sort_values(by='img_no',ascending=True)

        return x
    
    df = df.groupby(by='cluster', group_keys=False).apply(select_minimum)

    out_df = df[df['selected'] == 'yes']
    
    df = df[df['selected'] == 'no']

    def fill_rest(x:pd.DataFrame):
        rem_crops = n_crops - int(min_n_crops*n_clusters)
        n_easy = int(percent_easy/100*rem_crops)
        n_hard = int(percent_hard/100*rem_crops)
        
        if(n_easy + n_hard != rem_crops):
            n_easy += (rem_crops-(n_easy+n_hard))

        x = x.sort_values(by='cluster_dist',ascending=True)
        x.iloc[0:n_easy, x.columns.get_loc('selected')] = 'yes'

        x = x.sort_values(by='cluster_dist',ascending=False)
        x.iloc[0:n_hard, x.columns.get_loc('selected')] = 'yes'

        x = x.sort_values(by='img_no',ascending=True)

        return x

    df = fill_rest(df)
    df = df[df['selected'] == 'yes']

    out_df = pd.concat([out_df,df]).reset_index(drop=True)
    out_df.index = list(out_df.index)
    out_df.index.name = 'img_no'

    out_df.drop('selected',axis=1,inplace=True)

    return out_df
