from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from data_loader_scripts.download import cifar10_transforms
from fed_df_data_loader.split_standard import split_standard
from fed_df_data_loader.common import make_data_loader
import numpy as np

if __name__ == "__main__":    
    cifar10_testset = CIFAR10(root='./data/', train=False, download=True, transform=cifar10_transforms())
    
    fullDataLoader = DataLoader(cifar10_testset,batch_size=32, shuffle=False, num_workers=2)

    new_data_loaders = split_standard(fullDataLoader,n_splits=2)

    partition_len = len(new_data_loaders[1].dataset)
    print(partition_len)
    print(new_data_loaders[1].dataset[0][0].shape)
    print(new_data_loaders[1].dataset[0][1])

    random_preds = np.random.rand(partition_len,10)
    new_data_loader = make_data_loader(new_data_loaders[1],random_preds)
    print(len(new_data_loader.dataset))
    print(new_data_loader.dataset[0][0].shape)
    print(new_data_loader.dataset[0][1])

    
    