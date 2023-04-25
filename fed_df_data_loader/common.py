import torch
from typing import Optional, Callable, Any
from torchvision.datasets import VisionDataset
from torch.utils.data import DataLoader
from flwr.common import NDArray


class DistillDataset(VisionDataset):
    def __init__(self, root: str, data=None, targets=None, transforms: Optional[Callable] = None, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.data = data
        self.targets = targets
        if(root):
            self.data, self.targets = torch.load(root)

    def __getitem__(self, index: int) -> Any:
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self) -> int:
        assert(len(self.data) == len(self.targets))
        return len(self.data)


def make_data_loader(img_dataloader: DataLoader, preds: NDArray, batch_size: int = 32, n_workers: int = 0) -> DataLoader:
    preds_tensor = torch.tensor(preds)
    all_imgs = []
    for imgs, _ in img_dataloader:
        all_imgs.append(imgs)

    all_imgs = torch.cat(all_imgs, dim=0)
    new_dataset = DistillDataset(None, data=all_imgs, targets=preds_tensor)
    new_data_loader = DataLoader(
        new_dataset, batch_size=batch_size, num_workers=n_workers, pin_memory=True, shuffle=False)

    return new_data_loader
