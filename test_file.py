from models.cifar_resnet import cifar_resnet
import torch
from data_loader_scripts.download import download_dataset
from torch.utils.data import DataLoader
from strategy.tools.clipping import clip_logits
import torch.nn as nn
import time


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cresnet8 = cifar_resnet(variant=8).to(torch.device('cuda'))    
    print(f'model initialised')
    
    train_data_path, test_set = download_dataset(data_storage_path="./data",dataset_name="cifar10")
    train_loader = DataLoader(dataset=test_set, batch_size=1024, pin_memory=True)
    
    print(f'train loader created')
    
    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(params= cresnet8.parameters(), lr=0.01)
    
    
    print(f'Training network')
    start_time = time.time()
    for epoch in range(10):
        print(f'Epoch commenced : {epoch}')
        b_no = 0
        correct, total = 0,0
        epoch_loss = 0
        for imgs,labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # outputs = cresnet8(imgs)
            outputs = clip_logits(outputs=cresnet8(imgs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            b_no += 1

            total += labels.size(0)
            correct += (torch.max(outputs.detach(),1)[1] == labels).sum().item()

        epoch_loss /= b_no
        epoch_acc = correct/total

        print(f'Epoch {epoch}, loss = {epoch_loss}, acc = {epoch_acc}')
    
    end_time = time.time()
    print(f'Training finished')
    print(f'Seconds taken for code : {end_time-start_time}')
    




    