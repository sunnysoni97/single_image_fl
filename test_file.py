from models.cifar_resnet import cifar_resnet
import torch
from fed_df_data_loader.get_crops_dataloader import get_distill_imgloader
from strategy.clustering import cluster_embeddings, prune_clusters


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cresnet8 = cifar_resnet(variant=8).to(torch.device('cuda'))    
    print(f'model initialised')
    distill_dataloader = get_distill_imgloader('./data/single_img_crops/crops', dataset_name='cifar10', batch_size=1024)
    print(f'distillation loader loaded, length : {len(distill_dataloader.dataset)}')
    cluster_df, score = cluster_embeddings(dataloader=distill_dataloader, model=cresnet8, device=device, seed=42, n_clusters=100)
    print(f'embeddings clustered, score : {score}')
    prune_df = prune_clusters(cluster_df,n_crops=3000)
    print(f'embeddings pruned, length : {len(prune_df)}')



    