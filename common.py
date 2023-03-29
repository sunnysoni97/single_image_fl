import torch
import torch.nn as nn
from torchvision.models import ResNet
from torch.utils.data import DataLoader
from typing import Dict


def test_model(model: ResNet, test_loader: DataLoader, DEVICE: torch.device) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct, total, loss = 0, 0, 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = correct/total
    return {'test_loss': loss, 'test_acc': accuracy}
