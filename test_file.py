import argparse
from models.cifar_resnet import cifar_resnet
import torch
from fed_df_data_loader.get_crops_dataloader import get_distill_imgloader
from strategy.tools.clustering import cluster_embeddings, prune_clusters, calculate_tsne, visualise_tsne, visualise_clusters
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str,
                    default=str(Path.joinpath(Path(__file__).parent, 'data')))


if __name__ == "__main__":

    args = parser.parse_args()
    data_path = Path.joinpath(Path(args.input_dir),
                              'single_img_crops', 'crops')
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    cresnet8 = cifar_resnet(variant=8).to(torch.device('cuda'))
    print(f'model initialised')

    train_set = get_distill_imgloader(path_to_crops=data_path, batch_size=1024)
    print('distillation image loader loaded')
    print(f'length of dataset : {len(train_set.dataset)}')

    n_clusters = 10
    cluster_df, score = cluster_embeddings(
        train_set, model=cresnet8, device=device, n_clusters=n_clusters, seed=42)
    print(f'Clustering done, cluster score : {score}')
    print(f'Length of clustered df : {len(cluster_df)}')

    pruned_df = prune_clusters(cluster_df, n_crops=3000, heuristic="easy")
    print(f"Pruning done")

    tsne_df = calculate_tsne(cluster_df=pruned_df, device=device, n_cpu=12)
    print("tsne calculated")
    print(tsne_df.head())

    # label_metadata = None
    label_metadata = [f'cluster_{x}' for x in range(10)]
    with open('test_tsne.png', 'wb') as f:
        visualise_tsne(tsne_df=tsne_df, out_file=f,
                       label_metadata=label_metadata, n_classes=10)

    with open('test_clusters.png', 'wb') as f:
        visualise_clusters(cluster_df=tsne_df,
                           device=device, file=f, grid_size=10)
