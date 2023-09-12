from torchmetrics import F1Score, MetricCollection, Precision, Recall, HammingDistance, AveragePrecision


def get_metrics(**kwargs) -> MetricCollection:
    """Defines metrics for measuring model performance

    Returns:
        MetricCollection: collection of metrics that we would check
    """

    AP_params = {item: kwargs[item] for item in kwargs if item != 'threshold'}

    return MetricCollection(
        {
            'f1': F1Score(**kwargs),
            'precision': Precision(**kwargs),
            'recall': Recall(**kwargs),
            'hamming_dist': HammingDistance(**kwargs),
            'AP': AveragePrecision(**AP_params),
        }
    )