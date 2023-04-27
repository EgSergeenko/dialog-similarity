import json
import os
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Turn(object):
    actor: str
    utterance: str
    embedding: np.ndarray | None


@dataclass
class Dialog(object):
    dialog_id: str
    embedding: np.ndarray | None
    turns: list[Turn]

    def compute_embeddings(
        self,
        cache_dir: str,
        model: SentenceTransformer,
        model_name: str,
    ) -> None:
        document = ' '.join([turn.utterance for turn in self.turns])
        embedding_filepath = os.path.join(
            cache_dir, '{0}_{1}.npy'.format(model_name, self.dialog_id),
        )
        if os.path.isfile(embedding_filepath):
            self.embedding = np.load(embedding_filepath)
        else:
            self.embedding = model.encode([document])[0]
            np.save(embedding_filepath, self.embedding)

        for idx, turn in enumerate(self.turns):
            embedding_filepath = os.path.join(
                cache_dir,
                '{0}_{1}_{2}.npy'.format(
                    model_name, self.dialog_id, idx,
                ),
            )
            if os.path.isfile(embedding_filepath):
                embedding = np.load(embedding_filepath)
            else:
                embedding = model.encode([turn.utterance])[0]
                np.save(embedding_filepath, embedding)
            self.turns[idx].embedding = embedding


@dataclass
class DialogTriplet(object):
    anchor_dialog: Dialog
    dialog_1: Dialog
    dialog_2: Dialog
    label: int
    confidence_score: float

    def compute_embeddings(
        self,
        cache_dir: str,
        model: SentenceTransformer,
        model_name: str,
    ) -> None:
        self.anchor_dialog.compute_embeddings(cache_dir, model, model_name)
        self.dialog_1.compute_embeddings(cache_dir, model, model_name)
        self.dialog_2.compute_embeddings(cache_dir, model, model_name)


def dialog_from_dict(dialog_dict: dict) -> Dialog:
    turns = []
    if isinstance(dialog_dict['turns'], dict):
        for idx in range(len(dialog_dict['turns']['utterance'])):
            utterance = dialog_dict['turns']['utterance'][idx]
            actor = 'SYSTEM'
            if dialog_dict['turns']['speaker'][idx] == 0:
                actor = 'USER'
            turns.append(Turn(actor, utterance, None))
    else:
        for turn in dialog_dict['turns']:
            turns.append(
                Turn(turn['speaker'], turn['utterance'], None),
            )
    return Dialog(
        dialog_id=dialog_dict['dialogue_id'].split('.')[0],
        turns=turns,
        embedding=None,
    )


def dialog_from_file(dialog_filepath: str) -> Dialog:
    with open(dialog_filepath) as dialog_file:
        return dialog_from_dict(json.load(dialog_file))
