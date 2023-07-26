import torch
import torch.nn as nn
import numpy as np
from models import init_model, get_parameters
from data_loader_scripts.download import get_transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

def params_to_tensors(param_iterator) -> torch.Tensor:
    params = []
    for param in param_iterator:
        params.append(param.flatten())

    params = torch.cat(tensors=params)
    return params
    


if __name__ == "__main__":

    device = torch.device('cuda')
    dataset = CIFAR10(root='./data',train=True, transform=get_transforms("cifar10",True),download=True)
    dataloader = DataLoader(dataset=dataset,batch_size=1024)
    model = init_model(dataset_name="cifar10", model_name="resnet8")
    model.to(device)

    pred_loss = nn.CrossEntropyLoss(reduction='mean')
    def wt_loss(new_wt:torch.tensor, past_wt:torch.tensor, factor:float):
        loss = factor/2*(torch.sum(new_wt-past_wt))**2
        return loss
    
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=0.01)

    start_parameters = params_to_tensors(model.parameters()).to(device)
    
    epochs=4
    for i in range(epochs):
        p_loss = 0.0
        epoch_loss = 0.0
        batches = 0
        for imgs,labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            new_parameters = params_to_tensors(model.parameters()).to(device)
            cur_p_loss = pred_loss(out,labels)
            cur_wt_loss = wt_loss(new_parameters,start_parameters,1.0)
            loss = cur_p_loss + cur_wt_loss
            epoch_loss += loss.item()
            p_loss += cur_p_loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batches+=1
        
        print(f'Avg Epoch Loss : {epoch_loss/batches}')
        print(f'Avg Pred Loss : {p_loss/batches}')
        print(f'Epoch done')
    








   





