from torchmetrics import F1Score, MetricCollection, Precision, Recall


def get_metrics(**kwargs) -> MetricCollection:
    """Defines metrics for measuring model performance

    Returns:
        MetricCollection: collection of metrics that we would check
    """
    return MetricCollection(
        {
            'f1': F1Score(**kwargs),
            'precision': Precision(**kwargs),
            'recall': Recall(**kwargs),
        }
    )