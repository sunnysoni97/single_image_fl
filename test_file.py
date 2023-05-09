from fed_df_data_loader.split_standard import create_std_distill_loader
from collections import defaultdict

if __name__ == "__main__":
    test_dataloader = create_std_distill_loader(dataset_name='cifar10',storage_path='./data',n_images=1000)
    dataset = test_dataloader.dataset
    
    label_dict = defaultdict(int)
    for i in range(len(dataset)):
        label_dict[dataset[i][1].item()] += 1
    
    print(label_dict)
    