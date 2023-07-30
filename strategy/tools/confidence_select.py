import pandas as pd


def prune_confident_crops(cluster_df: pd.DataFrame, confidence_threshold: float = 0.1):

    # creating a new dataframe for selection
    new_df = cluster_df.copy()
    selected = ['yes' for i in range(len(new_df))]
    new_df['selected'] = selected

    # removing top k percentile of crops per class

    def remove_topk(x: pd.DataFrame):
        n = int(confidence_threshold*len(x))

        x = x.sort_values(by='conf', ascending=False)
        x.iloc[0:n, x.columns.get_loc('selected')] = 'no'

        return x

    new_df = new_df.groupby(by='pred', group_keys=False).apply(remove_topk)

    new_df = new_df[new_df['selected'] == 'yes']
    new_df.drop('selected', axis=1, inplace=True)
    new_df.sort_index(inplace=True)

    return new_df
