from data_loader import load_dataset
from models import init_model
import torch

if __name__ == "__main__":
    t_loaders, v_loaders, te_loader = load_dataset("cifar10", 5, 32, 42)
    resnet18 = init_model("resnet18", 10)

    print(len(t_loaders[0]))
    print(len(v_loaders))
    print("---")
    print(len(t_loaders[0].dataset))
    print(len(v_loaders[0].dataset))
    print(len(te_loader.dataset))
    print("---")

    i = 0
    resnet18.eval()
    for (images, labels) in t_loaders[1]:
        if(i == 5):
            break
        # print(f'One hot true label : {torch.nn.functional.one_hot(labels[0],10)}')
        pred = resnet18.forward(images)
        true_value = labels
        print(f'Prediction : {torch.argmax(pred[0])}, Label : {true_value[0]}')
        print(
            f'Loss : {torch.nn.functional.cross_entropy(input=pred,target=true_value,reduction="sum")}')
        i += 1
