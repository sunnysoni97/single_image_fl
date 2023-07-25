import torch
from models import init_model
from fed_df_data_loader.get_crops_dataloader import get_distill_imgloader
from strategy.tools.clustering import cluster_embeddings, prune_clusters, prepare_for_transport, extract_from_transport
from strategy.tools.confidence_select import prune_confident_crops

if __name__ == "__main__":

    device = torch.device('cuda')
    model = init_model(dataset_name="cifar10", model_name="resnet8")

    dl = get_distill_imgloader('./data/single_img_crops/crops',batch_size=1024)
    print(len(dl.dataset))

    embed_df, _ = cluster_embeddings(dataloader=dl, model=model, device=device, n_clusters=20)
    embed_df = prune_clusters(raw_dataframe=embed_df, n_crops=10000, heuristic="easy")

    print(len(embed_df))

    embed_df = prune_confident_crops(model=model, device=device, cluster_df=embed_df, confidence_threshold=0.2)

    print(len(embed_df))
    print(embed_df.head())

    transport_in = prepare_for_transport(embed_df)
    transport_out = extract_from_transport(transport_in)

    print(len(transport_out.dataset))






