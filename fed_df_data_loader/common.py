import torch
from typing import Optional, Callable, Any
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from flwr.common import NDArray

import torchvision.transforms as transforms


class DistillDataset(VisionDataset):
    def __init__(self, root: str, data=None, targets=None, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.data = data
        self.targets = targets
        if (root):
            self.data, self.targets = torch.load(root)

    def __getitem__(self, index: int) -> Any:
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self) -> int:
        assert (len(self.data) == len(self.targets))
        return len(self.data)


def make_data_loader(img_dataloader: DataLoader, preds: NDArray, batch_size: int = 32, n_workers: int = 0) -> DataLoader:
    preds_tensor = torch.tensor(preds)
    all_imgs = []
    for imgs, _ in img_dataloader:
        all_imgs.append(imgs)

    all_imgs = torch.cat(all_imgs, dim=0)
    new_dataset = DistillDataset(None, data=all_imgs, targets=preds_tensor)
    new_data_loader = DataLoader(
        new_dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=False, shuffle=True)

    return new_data_loader


def get_distill_transforms(tgt_dataset: str = "cifar10", transform_type: str = "v0"):
    implemented_dataset = ["cifar10", "cifar100",
                           "pathmnist", "pneumoniamnist", "organamnist"]
    if (not tgt_dataset in implemented_dataset):
        raise NotImplementedError(
            f"{tgt_dataset} has not been implemented for distillation transforms!")

    if (tgt_dataset == "cifar10"):
        t_normalise = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
    elif (tgt_dataset == "cifar100"):
        t_normalise = transforms.Compose([
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])
    else:
        t_normalise = transforms.Compose([
            transforms.Normalize([0.5], [0.5])
        ])
        if (tgt_dataset in ['pneumoniamnist', 'organamnist']):
            t_normalise.transforms.extend([transforms.Grayscale()])

    if (transform_type == "v0"):
        t_initial = transforms.Compose([])
    elif (transform_type == "v1"):
        t_initial = transforms.Compose([
            transforms.RandomCrop((32, 32), 4)
        ])
    else:
        raise NotImplementedError(
            f'{transform_type} not implemented yet for distillation transforms!')

    t_compose = transforms.Compose(
        t_initial.transforms + [transforms.ToTensor()] + t_normalise.transforms
    )

    return t_compose
