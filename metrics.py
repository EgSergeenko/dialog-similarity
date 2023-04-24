from abc import ABC, abstractmethod

import numpy as np
import scipy
from sklearn.metrics import accuracy_score

from dialog import Dialog, DialogTriplet


class BaseMetric(ABC):
    def __init__(self, is_inverted: bool) -> None:
        self.is_inverted = is_inverted

    @abstractmethod
    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        ...


class ExampleMetric(BaseMetric):
    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        return 1.0


class AvgEmbeddingDistance(BaseMetric):
    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        embedding_1 = self._get_avg_embedding(dialog_1)
        embedding_2 = self._get_avg_embedding(dialog_2)
        return scipy.spatial.distance.cosine(embedding_1, embedding_2)

    def _get_avg_embedding(self, dialog: Dialog) -> np.ndarray:
        embedding = dialog.turns[0].embedding
        for i in range(1, len(dialog.turns)):
            embedding += dialog.turns[i].embedding
        return embedding / len(dialog.turns)


class ConversationalEditDistance(BaseMetric):
    def __init__(
        self,
        is_inverted: bool,
        insertion_weight: float = 1.0,
        deletion_weight: float = 1.0,
        substitution_weight: float = 2.2,
    ) -> None:
        super().__init__(is_inverted)
        self.insertion_weight = insertion_weight
        self.deletion_weight = deletion_weight
        self.substitution_weight = substitution_weight

    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        n, m = len(dialog_1.turns), len(dialog_2.turns)
        distances = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            distances[i][0] = distances[i - 1][0] + self.deletion_weight

        for j in range(1, m + 1):
            distances[0][j] = distances[0][j - 1] + self.insertion_weight

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                insertion_cost = distances[i][j - 1] + self.insertion_weight

                deletion_cost = distances[i - 1][j] + self.deletion_weight

                substitution_cost = np.inf

                if dialog_1.turns[i - 1].actor == dialog_2.turns[j - 1].actor:
                    cosine_distance = scipy.spatial.distance.cosine(
                        dialog_1.turns[i - 1].embedding,
                        dialog_2.turns[j - 1].embedding,
                    )
                    cosine_distance *= self.substitution_weight
                    substitution_cost = distances[i - 1][j - 1] + cosine_distance

                distances[i, j] = min(
                    insertion_cost,
                    deletion_cost,
                    substitution_cost,
                )

        return distances[n][m]


def get_metric_agreement(
    dialog_triplets: list[DialogTriplet],
    metric: BaseMetric,
    confidence_threshold: float,
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
        if metric.is_inverted:
            prediction = ~prediction
        predictions.append(prediction)

    return accuracy_score(labels, predictions)
