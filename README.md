# Federated Learning with a Single Shared Image

(Original paper accepted in CVPR'24 [[LIMIT](https://hirokatsukataoka16.github.io/CVPR-2024-LIMIT/)] workshop proceedings)

![Schematic figure of Federated Learning with a Single Shared Image](splash_fig.png)

## Abstract

Federated Learning (FL) enables multiple machines to collaboratively train a machine learning model without sharing of private training data. Yet, especially for heterogeneous models, a key bottleneck remains the transfer of knowledge gained from each client model with the server. One popular method, FedDF, uses distillation to tackle this task with the use of a common, shared dataset on which predictions are exchanged. However, in many contexts such a dataset might be difficult to acquire due to privacy and the clients might not allow for storage of a large shared dataset. To this end, in this paper, we introduce a new method that improves this knowledge distillation method to only rely on a single shared image between clients and server. In particular, we propose a novel adaptive dataset pruning algorithm that selects the most informative crops generated from only a single image. With this, we show that federated learning with distillation under a limited shared dataset budget works better by using a single image compared to multiple individual ones. Finally, we extend our approach to allow for training heterogeneous client architectures by incorporating a non-uniform distillation schedule and client-model mirroring on the server side.

## In This Repository

To conduct our research and experimentation, we have made use of [Flower Framework](https://github.com/adap/flower) alongside vanilla PyTorch and required libraries in this project.

`simulate.py` is the entrypoint code, which begins the experiment and finishes with the output of the results according to your experiment settings.

There's a large list of associated arguments which can be passed to setup your experiment the way you want to. The description and their usage has been integrated in the argument parser itself, which can be accessed with `python simulate.py --help`

Alternatively, have a look at `local_simulate.sh` bash script for an example on how to conduct an experiment using our code.

### Setting up the conda environment

Using `environment.yml` , a new conda environment can be created with all the required python libraries used during the development of this project.

### Basic Steps for conducting an experiment

1. Produce the single image crops using `make_single_img_dataset.py`
2. Conduct federated learning experiments using `simulate.py`
3. The results are output in a text file in the output directory at the end of simulation.

#### Important Fix for fixing GPU memory leak (flwr simulation)

1. Find the `ray_client_proxy.py` in your local pkg installation directory of python packages under the folder `"flwr/simulation/ray_transport/"`.
2. Replace `@ray.remote()` before `'launch_and_XXX'` methods to `@ray.remote(max_calls=1)`
3. Save the file and execute the main simulation python script of our project as usual from now on.

## Reference 
```
@inproceedings{soni2024federated,
  title={Federated Learning with a Single Shared Image},
  author={Sunny Soni and Aaqib Saeed and Yuki M. Asano},
  booktitle={CVPR Workshop Proceedings},
  year={2024},
}
```
