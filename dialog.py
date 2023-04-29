import json
import os
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Act(object):
    intent: str
    slot_type: str


@dataclass
class Turn(object):
    actor: str
    utterance: str
    embedding: np.ndarray | None
    acts: list[Act]

    def acts_to_string(self) -> str:
        intent_slot_mapping = OrderedDict()
        for act in self.acts:
            if act.intent not in intent_slot_mapping:
                intent_slot_mapping[act.intent] = []
            intent_slot_mapping[act.intent].append(act.slot_type)

        intent_strings = []
        for intent, slots in intent_slot_mapping.items():
            slots_string = '_'.join(sorted(slots))
            intent_string = '{0}_{1}'.format(intent, slots_string)
            intent_strings.append(intent_string.strip('_'))
        return '_'.join(intent_strings)


@dataclass
class Dialog(object):
    dialog_id: str
    embedding: np.ndarray | None
    turns: list[Turn]
    services: list[str]

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
            turns.append(
                Turn(
                    actor=actor,
                    utterance=utterance,
                    embedding=None,
                    acts=[],
                ),
            )
    else:
        for turn in dialog_dict['turns']:
            acts = []
            for frame in turn['frames']:
                acts.extend([*frame['actions']])
            turns.append(
                Turn(
                    actor=turn['speaker'],
                    utterance=turn['utterance'],
                    embedding=None,
                    acts=[Act(act['act'], act['slot']) for act in acts],
                ),
            )
    return Dialog(
        dialog_id=dialog_dict['dialogue_id'].split('.')[0],
        turns=turns,
        embedding=None,
        services=dialog_dict['services'],
    )


def dialog_from_file(dialog_filepath: str) -> Dialog:
    with open(dialog_filepath) as dialog_file:
        return dialog_from_dict(json.load(dialog_file))
