from data_loader import load_dataset
from models import init_model
import torch

if __name__ == "__main__":
    t_loaders, v_loaders, te_loader = load_dataset("cifar10",5,32,42)
    resnet18 = init_model("resnet18",10)

    print(len(t_loaders))
    print(len(v_loaders))
    print("---")
    print(len(t_loaders[0].dataset))
    print(len(v_loaders[0].dataset))
    print(len(te_loader.dataset))
    print("---")

    i = 0
    resnet18.eval()
    for (images,labels) in te_loader:
        if(i==5):
            break
        print(f'Prediction : {torch.argmax(resnet18.forward(images[0].unsqueeze(0)))}, Label : {labels[0]}')
        i+=1