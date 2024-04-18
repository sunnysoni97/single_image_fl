# Federated Learning with a Single Shared Image

(Original paper accepted in CVPR'24 [[LIMIT](https://hirokatsukataoka16.github.io/CVPR-2024-LIMIT/)] workshop proceedings)

![Schematic figure of Federated Learning with a Single Shared Image](splash_fig.png)

## Abstract

Federated Learning (FL) enables multiple machines to collaboratively train a machine learning model without sharing of private training data. Yet, especially for heterogeneous models, a key bottleneck remains the transfer of knowledge gained from each client model with the server. One popular method, FedDF, uses distillation to tackle this task with the use of a common, shared dataset on which predictions are exchanged. However, in many contexts such a dataset might be difficult to acquire due to privacy and the clients might not allow for storage of a large shared dataset. To this end, in this paper, we introduce a new method that improves this knowledge distillation method to only rely on a single shared image between clients and server. In particular, we propose a novel adaptive dataset pruning algorithm that selects the most informative crops generated from only a single image. With this, we show that federated learning with distillation under a limited shared dataset budget works better by using a single image compared to multiple individual ones. Finally, we extend our approach to allow for training heterogeneous client architectures by incorporating a non-uniform distillation schedule and client-model mirroring on the server side.

## Results at a Glance

The following table shows test evaluation comparison using CIFAR10 dataset for private data emulation using ResNet-8 architecture during 30 rounds of federated training, using our method against an existing naive approach. [Full results in the official paper]

| Shared Dataset                        | No. of Shared Pixels | Accuracy |
| ------------------------------------- | -------------------- | :------: |
| CIFAR100 Test Samples (500)           | 0.5 Million          |  73.3 %  |
| CIFAR100 Test Samples (5000)          | 5 Million            |  75.7 %  |
| **Single Image with Patch Selection** | 0.5 Million          |  76.4 %  |

## Project Implementation

To conduct our research and experimentation, we have made use of [Flower Framework](https://github.com/adap/flower) alongside vanilla [PyTorch](https://github.com/pytorch/pytorch) and required libraries in this project.

`simulate.py` is the entrypoint code, which begins the experiment and finishes with the output of the results according to your experiment settings.

There's a large list of associated arguments which can be passed to setup your experiment the way you want to. The description and their usage has been integrated in the argument parser itself, which can be accessed with `python simulate.py --help`

Alternatively, have a look at `local_simulate.sh` bash script for an example on how to conduct an experiment using our code.

### Setting up the environment and Running experiments

Using `environment.yml` , a new conda environment can be created with all the required python libraries used during the development of this project.

#### Conducting an experiment

1. Produce the single image crops using `make_single_img_dataset.py`. Refer to the `--help` argument to find the complete list of arguments passable to the script.
2. Conduct federated learning experiments using `simulate.py` using your settings.
3. The results are output on the console which can be written to a text file using your preferred bash procedure.

#### Work-around for fixing GPU memory leak (flwr simulation)

1. Find the `ray_client_proxy.py` in your local pkg installation directory of python packages under the folder `"flwr/simulation/ray_transport/"`.
2. Replace `@ray.remote()` before `'launch_and_XXX'` methods to `@ray.remote(max_calls=1)`
3. Save the file and execute the main simulation python script of our project as usual from now on.

### Supported Model Architectures and Datasets

#### Model Architectures

Our project supports the following model architectures for client and global models during simulation: ResNet [[Paper](https://arxiv.org/abs/1512.03385)], Wide-Resnet [[Paper](https://arxiv.org/abs/1605.07146)].

There's no pretraining involved and we train the models from scratch during the simulation.

#### Datasets

Our project supports the following public datasets for emulating private data and evaluation purposes during simulation: CIFAR10/100 [[Link](https://www.cs.toronto.edu/~kriz/cifar.html)]; MedMNIST [[Link](https://github.com/MedMNIST/MedMNIST)] subset including PneumoniaMNIST, PathMNIST and OrganAMNIST.

## Reference

```
@inproceedings{soni2024federated,
  title={Federated Learning with a Single Shared Image},
  author={Sunny Soni and Aaqib Saeed and Yuki M. Asano},
  booktitle={CVPR Workshop Proceedings},
  year={2024},
}
```
