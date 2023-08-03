import pandas as pd
import numpy as np


def prune_confident_crops(cluster_df: pd.DataFrame, confidence_threshold: float = 0.1, confidence_strategy: str = "top"):

    # checking for valid value of confidence strategy

    if (confidence_strategy not in ["top", "bottom", "random"]):
        raise ValueError(
            f'{confidence_strategy} is not valid value for entropy selection strategy!')

    # creating a new dataframe for selection
    new_df = cluster_df.copy()
    selected = ['yes' for i in range(len(new_df))]
    new_df['selected'] = selected

    # removing top k percentile of crops per class

    def remove_k(x: pd.DataFrame, heuristic: str):
        n = int(confidence_threshold*len(x))

        if (heuristic == "top" or heuristic == "bottom"):

            if (heuristic == "top"):
                x = x.sort_values(by='conf', ascending=False)
            else:
                x = x.sort_values(by='conf', ascending=True)

            x.iloc[0:n, x.columns.get_loc('selected')] = 'no'

        else:
            id_list = np.random.randint(low=0, high=len(x), size=n)
            x.iloc[id_list, x.columns.get_loc('selected')] = "no"

        return x

    new_df = new_df.groupby(by='pred', group_keys=False).apply(
        remove_k, heuristic=confidence_strategy)

    new_df = new_df[new_df['selected'] == 'yes']
    new_df.drop('selected', axis=1, inplace=True)
    new_df.sort_index(inplace=True)

    return new_df
