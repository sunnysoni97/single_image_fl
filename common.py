import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List
import numpy as np
from models import init_model, set_parameters


def test_model(model_name: str, model_n_classes: int, parameters: List[np.ndarray], test_loader: DataLoader, DEVICE: torch.device) -> Dict[str, float]:
    model = init_model(model_name, model_n_classes)
    set_parameters(model, parameters)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct, total, loss = 0, 0, 0.0
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.detach(), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = correct/total
    return {'test_loss': loss, 'test_acc': accuracy}
