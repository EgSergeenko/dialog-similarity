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


def dialog_from_dict(dialog_dict):
    turns = []
    for turn in dialog_dict['turns']:
        turns.append(
            Turn(turn['speaker'], turn['utterance'], None),
        )
    return Dialog(
        dialog_id=dialog_dict['dialogue_id'],
        turns=turns,
    )
