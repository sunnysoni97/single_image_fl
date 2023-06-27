from PIL import Image
from data_loader_scripts.create_dataloader import TorchVision_FL
from fed_df_data_loader.common import get_distill_transforms
import os
import numpy as np
from torch.utils.data import DataLoader
import copy


def get_distill_imgloader(path_to_crops: os.PathLike, dataset_name: str = "cifar10", batch_size: int = 32, num_workers: int = 0, distill_transforms: str = "v0") -> DataLoader:
    path = path_to_crops
    files = [os.path.join(path, f) for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))]

    image_list = []
    for filename in files:
        im = copy.deepcopy(Image.open(filename))
        image_list.append(im)

    no_of_images = len(image_list)
    dummy_targets = np.zeros((no_of_images, 1))

    transformF = get_distill_transforms(
        tgt_dataset=dataset_name, transform_type=distill_transforms)

    new_dataset = TorchVision_FL(
        data=image_list, targets=dummy_targets, transform=transformF)

    kwargs = {"batch_size": batch_size, "drop_last": False,
              "num_workers": num_workers, "pin_memory": True, "shuffle": False}
    distill_img_loader = DataLoader(new_dataset, **kwargs,)

    return distill_img_loader
