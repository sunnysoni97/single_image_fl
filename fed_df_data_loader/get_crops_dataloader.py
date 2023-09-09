from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader
import multiprocessing as mp
from joblib import Parallel, delayed, parallel_backend
from typing import Tuple

from data_loader_scripts.create_dataloader import TorchVision_FL
from fed_df_data_loader.common import get_distill_transforms


def __fetch_files(file_list: list, thread_no: int) -> Tuple[list, int]:
    im_list = []

    for file_name in file_list:
        with Image.open(file_name) as file:
            im = file.copy()
        im_list.append(im)

    return im_list, thread_no


def get_distill_imgloader(path_to_crops: os.PathLike, dataset_name: str = "cifar10", batch_size: int = 32, num_workers: int = 0, distill_transforms: str = "v0") -> DataLoader:

    path = path_to_crops
    files = [os.path.join(path, f) for f in sorted(os.listdir(
        path)) if os.path.isfile(os.path.join(path, f))]

    no_threads = num_workers
    if (num_workers == 0):
        no_threads = mp.cpu_count()

    files_split = np.array_split(np.array(files), no_threads)

    with parallel_backend(backend='multiprocessing', n_jobs=no_threads):
        images_split = Parallel()(delayed(__fetch_files)(
            files_split[i], i) for i in range(no_threads))

    images_split = sorted(images_split, key=lambda x: x[1])

    image_list = []
    for img_list_thread, _ in images_split:
        image_list.extend(img_list_thread)

    no_of_images = len(image_list)
    dummy_targets = np.zeros((no_of_images, 1))

    transformF = get_distill_transforms(
        tgt_dataset=dataset_name, transform_type=distill_transforms)

    new_dataset = TorchVision_FL(
        data=image_list, targets=dummy_targets, transform=transformF)

    kwargs = {"batch_size": batch_size, "drop_last": False,
              "num_workers": num_workers, "pin_memory": True, "shuffle": False}
    distill_img_loader = DataLoader(new_dataset, **kwargs)

    return distill_img_loader
