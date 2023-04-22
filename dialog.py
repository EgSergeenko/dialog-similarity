import json
from dataclasses import dataclass

import torch


@dataclass
class Turn(object):
    actor: str
    utterance: str
    embedding: torch.Tensor | None


@dataclass
class Dialog(object):
    dialog_id: str
    turns: list[Turn]


@dataclass
class DialogTriplet(object):
    anchor_dialog: Dialog
    dialog_1: Dialog
    dialog_2: Dialog
    label: int
    confidence_score: float


def dialog_from_dict(dialog_dict: dict) -> Dialog:
    turns = []
    for turn in dialog_dict['turns']:
        turns.append(
            Turn(turn['speaker'], turn['utterance'], None),
        )
    return Dialog(
        dialog_id=dialog_dict['dialogue_id'],
        turns=turns,
    )


def dialog_from_file(dialog_filepath: str) -> Dialog:
    with open(dialog_filepath) as dialog_file:
        return dialog_from_dict(json.load(dialog_file))
