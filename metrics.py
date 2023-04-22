from typing import Callable

from sklearn.metrics import accuracy_score

from dialog import Dialog, DialogTriplet


class ExampleMetric(object):
    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        # computing ...
        return 1.0


def get_metric_agreement(
    dialog_triplets: list[DialogTriplet],
    metric: Callable,
    confidence_threshold: float,
    inverted_metric: bool = False,
) -> float:
    labels, predictions = [], []

    for dialog_triplet in dialog_triplets:
        if dialog_triplet.confidence_score < confidence_threshold:
            continue
        labels.append(dialog_triplet.label)
        score_1 = metric(
            dialog_triplet.anchor_dialog, dialog_triplet.dialog_1,
        )
        score_2 = metric(
            dialog_triplet.anchor_dialog, dialog_triplet.dialog_2,
        )
        prediction = 1
        if score_1 < score_2:
            prediction = 0
        if inverted_metric:
            prediction = ~prediction
        predictions.append(prediction)

    return accuracy_score(labels, predictions)
