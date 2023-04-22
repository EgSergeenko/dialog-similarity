from sklearn.metrics import accuracy_score


class ExampleMetric(object):
    def __call__(self, dialog_1, dialog_2):
        # computing ...
        return 1.0


def get_metric_agreement(
    dialog_triplets, metric, confidence_threshold, inverted_metric=False,
):
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
