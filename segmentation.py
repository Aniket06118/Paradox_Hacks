# segmentation.py

from metric import compute_metrics


def segment_by_column(df, column_name):
    results = {}

    for value, group in df.groupby(column_name):
        results[value] = compute_metrics(group)

    return results
