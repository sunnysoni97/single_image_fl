from torchvision.datasets import VisionDataset
from typing import Callable, Optional, Tuple, Any
import torch
from PIL import Image
from pathlib import Path
import numpy as np
from data_loader_scripts.download import cifar10_transforms, cifar100_transforms
from torch.utils.data import DataLoader


class TorchVision_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
        self,
        path_to_data=None,
        data=None,
        targets=None,
        transform: Optional[Callable] = None,
    ) -> None:
        path = path_to_data.parent if path_to_data else None
        super(TorchVision_FL, self).__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not isinstance(img, Image.Image):  # if not PIL image
            if not isinstance(img, np.ndarray):  # if torch tensor
                img = img.numpy()

            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def get_dataset(dataset_name: str, path_to_data: Path, cid: str, partition: str):
    # generate path to cid's data
    path_to_data = path_to_data / cid / (partition + ".pt")
    if(dataset_name == "cifar10"):
        transformF = cifar10_transforms()
    elif(dataset_name == "cifar100"):
        transformF = cifar100_transforms()
    else:
        raise ValueError(
            f"Can't get this dataset : {dataset_name} not defined yet!")

    return TorchVision_FL(path_to_data, transform=transformF)


def create_dataloader(
    dataset_name: str, path_to_data: str, cid: str, is_train: bool, batch_size: int, workers: int
):
    """Generates trainset/valset object and returns appropiate dataloader."""

    partition = "train" if is_train else "val"
    dataset = get_dataset(dataset_name, Path(path_to_data), cid, partition)

    # we use as number of workers all the cpu cores assigned to this actor
    kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
    return DataLoader(dataset, batch_size=batch_size, **kwargs)
