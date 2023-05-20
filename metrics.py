from abc import ABC, abstractmethod

import numpy as np
import scipy
from sklearn.metrics import accuracy_score

from dialog import Dialog, DialogTriplet
from collections import defaultdict

from tabulate import tabulate


class BaseMetric(ABC):
    def __init__(self, is_inverted: bool) -> None:
        self.is_inverted = is_inverted

    @abstractmethod
    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        ...


class ExampleMetric(BaseMetric):
    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        return 1.0


class EmbeddingMetric(BaseMetric):
    def __init__(self, is_inverted: bool, embedding_type: str):
        super().__init__(is_inverted)
        if embedding_type not in {'dialog', 'turn'}:
            raise NotImplementedError()
        if embedding_type == 'dialog':
            self.get_embedding = self.get_dialog_embedding
        elif embedding_type == 'turn':
            self.get_embedding = self.get_average_turn_embedding
        else:
            raise NotImplementedError()

    @abstractmethod
    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        ...

    def get_average_turn_embedding(self, dialog: Dialog) -> np.ndarray:
        embedding = dialog.turns[0].embedding
        for i in range(1, len(dialog.turns)):
            embedding += dialog.turns[i].embedding
        return embedding / len(dialog.turns)

    def get_dialog_embedding(self, dialog: Dialog) -> np.ndarray:
        return dialog.embedding


class CosineDistance(EmbeddingMetric):
    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        return scipy.spatial.distance.cosine(
            self.get_embedding(dialog_1),
            self.get_embedding(dialog_2),
        )


class LpDistance(EmbeddingMetric):
    def __init__(
        self, is_inverted: bool, embedding_type: str, p: int,
    ) -> None:
        super().__init__(is_inverted, embedding_type)
        self.p = p

    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        return scipy.spatial.distance.minkowski(
            self.get_embedding(dialog_1),
            self.get_embedding(dialog_2),
            p=self.p,
        )


class DotProductSimilarity(EmbeddingMetric):
    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        return self.get_embedding(dialog_1) @ self.get_embedding(dialog_2)


class EditDistance(BaseMetric):
    def __init__(
        self,
        is_inverted: bool,
        normalize: bool,
        insertion_weight: float = 1.0,
        deletion_weight: float = 1.0,
        substitution_weight: float = 2.0,
    ) -> None:
        super().__init__(is_inverted)
        self.normalize = normalize
        self.insertion_weight = insertion_weight
        self.deletion_weight = deletion_weight
        self.substitution_weight = substitution_weight

    @abstractmethod
    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        ...


class ConversationalEditDistance(EditDistance):
    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        n, m = len(dialog_1.turns), len(dialog_2.turns)
        distances, _ = self._compute_distance_matrix(dialog_1, dialog_2, n, m)
        if self.normalize:
            return distances[n][m] / max(n, m)
        return distances[n][m]

    def visualize(
        self, dialog_1: Dialog, dialog_2: Dialog, output_filepath: str,
    ) -> None:
        n, m = len(dialog_1.turns), len(dialog_2.turns)
        distances, actions = self._compute_distance_matrix(dialog_1, dialog_2, n, m)
        action_list = self._get_actions_list(actions, n, m)
        action_list.reverse()
        i, j = 0, 0
        dialog_1_list = []
        dialog_2_list = []

        for action in action_list:
            if action == "I":
                dialog_1_list.append("Insertion \n\n")
                dialog_2_list.append(f"{dialog_2.turns[j].actor}:{dialog_2.turns[j].utterance} \n\n")
                dialog_1_list.append("-"*20)
                dialog_2_list.append("-"*20)
                j += 1

            if action == "D":
                dialog_2_list.append("Deletion")
                dialog_1_list.append(f"{dialog_1.turns[i].actor}:{dialog_1.turns[i].utterance} \n\n")
                dialog_1_list.append("-" * 20)
                dialog_2_list.append("-" * 20)
                i += 1

            if action == "S":
                dialog_1_list.append(f"{dialog_1.turns[i].actor}:{dialog_1.turns[i].utterance} \n\n")
                dialog_2_list.append(f"{dialog_2.turns[j].actor}:{dialog_2.turns[j].utterance} \n\n")
                dialog_1_list.append("-" * 20)
                dialog_2_list.append("-" * 20)
                j += 1
                i += 1

        table = [[x, y] for x, y in zip(dialog_1_list, dialog_2_list)]
        html_str = tabulate(table, tablefmt='html')
        with open(output_filepath, 'w') as file:
            file.write(html_str)

    def _get_actions_list(self, actions: dict, n: int, m: int) -> list:
        actions_list = []
        i, j = n, m
        while i > 0 and j > 0:
            last_action = actions[i][j]
            actions_list.append(actions[i][j])
            if last_action == "I":
                j -= 1
                continue
            if last_action == "D":
                i -= 1
                continue
            if last_action == "S":
                i -= 1
                j -= 1
                continue
        return actions_list

    def _compute_distance_matrix(
            self, dialog_1: Dialog, dialog_2: Dialog, n: int, m: int,
    ) -> (np.ndarray, dict):
        actions_table = defaultdict(dict)
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
                actions = np.array([insertion_cost, deletion_cost, substitution_cost])
                actions_word = ["I", "D", "S"]
                index_min = np.argmin(actions)
                actions_table[i][j] = actions_word[index_min]
                distances[i, j] = actions[index_min]
        return distances, actions_table


class StructuralEditDistance(EditDistance):
    def __init__(
        self,
        is_inverted: bool,
        normalize: bool,
        insertion_weight: float = 1.0,
        deletion_weight: float = 1.0,
        substitution_weight: float = 2.0,
        transpositions: bool = False,
    ) -> None:
        super().__init__(
            is_inverted,
            normalize,
            insertion_weight,
            deletion_weight,
            substitution_weight,
        )
        self.transpositions = transpositions

    def __call__(self, dialog_1: Dialog, dialog_2: Dialog) -> float:
        n, m = len(dialog_1.turns), len(dialog_2.turns)
        distances = self._compute_distance_matrix(dialog_1, dialog_2, n, m)
        if self.normalize:
            return distances[n][m] / max(n, m)
        return distances[n][m]

    def _compute_distance_matrix(
        self,
        dialog_1: Dialog,
        dialog_2: Dialog,
        n: int,
        m: int
    ) -> np.ndarray:
        distances = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            distances[i][0] = distances[i - 1][0] + self.deletion_weight

        for j in range(1, m + 1):
            distances[0][j] = distances[0][j - 1] + self.insertion_weight

        sigma = set()

        sigma.update([turn.acts_string for turn in dialog_1.turns])
        sigma.update([turn.acts_string for turn in dialog_2.turns])

        last_left_t = {c: 0 for c in sigma}

        for i in range(1, n + 1):
            last_right_buf = 0
            for j in range(1, m + 1):
                dialog_1_acts = dialog_1.turns[i - 1].acts_string
                dialog_2_acts = dialog_2.turns[j - 1].acts_string

                last_left = last_left_t[dialog_2_acts]
                last_right = last_right_buf

                if dialog_1_acts == dialog_2_acts:
                    last_right_buf = j

                insertion_cost = distances[i][j - 1] + self.insertion_weight

                deletion_cost = distances[i - 1][j] + self.deletion_weight

                substitution_cost = distances[i - 1][j - 1]
                if dialog_1_acts != dialog_2_acts:
                    substitution_cost += self.substitution_weight

                transposition_cost = substitution_cost + 1
                if self.transpositions and last_left > 0 and last_right > 0:
                    transposition_cost = distances[last_left - 1][last_right - 1]
                    transposition_cost += i - last_left + j - last_right - 1

                distances[i, j] = min(
                    insertion_cost,
                    deletion_cost,
                    substitution_cost,
                    transposition_cost,
                )

                last_left_t[dialog_1_acts] = i

        return distances


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
            prediction = 1 - prediction
        predictions.append(prediction)

    return accuracy_score(labels, predictions)
