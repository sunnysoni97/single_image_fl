from flwr.server.strategy.aggregate import aggregate
import numpy as np
import torch


if __name__ == "__main__":
    a = [[0,0,0,0],[0,0,0,0]]
    b = [[10,10,10,10],[20,20,20,20]]
    results = [(np.array(a),500),(np.array(b),500)]
    avg = np.array(aggregate(results))
    print("before")
    print(avg)
    avg = torch.tensor(avg)
    print("after")
    print(avg)