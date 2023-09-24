from models import init_model
import torch


if __name__ == "__main__":
    print("Testing shape")
    dataset_name = "cifar10"
    model_name = "wresnet16-4"
    # model_name = "resnet8"

    a = init_model(dataset_name=dataset_name, model_name=model_name)

    t = torch.randn((1, 3, 32, 32))

    feats = a.forward_avgpool(t)
    cls = a.forward(t)

    print(feats.size())
    print(cls.size())

    print("Success")
