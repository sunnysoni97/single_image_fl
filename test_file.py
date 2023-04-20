from flwr.server.strategy.aggregate import aggregate
import numpy as np

if __name__ == "__main__":
    no_preds = 3
    preds = np.random.rand(no_preds,10)
    preds2 = np.random.rand(no_preds,10)
    preds_list = [(preds,no_preds),(preds2,no_preds)]
    aggregated_preds = aggregate(preds_list)
    aggregated_preds = np.array(aggregated_preds)
    print(aggregated_preds)