import json
import os
from copy import deepcopy

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset
from tqdm import tqdm

from dialog import Dialog, dialog_from_dict
from model import Embedder


class TranslationModel(object):
    def __init__(
        self,
        model_name: str,
        device: str,
        num_beams: int = 3,
        num_return_utterances: int =1,
    ) -> None:
        if not torch.cuda.is_available() and 'cuda' in device:
            device = 'cpu'
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            self.device,
        )
        self.num_beams = num_beams
        self.num_return_utterances = num_return_utterances

    def __call__(self, texts: list[str]) -> list[str]:
        tokens = self.tokenizer(texts, return_tensors='pt', padding=True)
        outputs = self.model.generate(
            input_ids=tokens['input_ids'].to(self.device),
            attention_mask=tokens['attention_mask'].to(self.device),
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_utterances,
        )
        return self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True,
        )


class AugmentationModel(object):
    def __init__(self, device: str):
        self.en_de_model = TranslationModel(
            'Helsinki-NLP/opus-mt-en-de', device,
        )
        self.de_en_model = TranslationModel(
            'Helsinki-NLP/opus-mt-de-en', device,
        )

    def __call__(self, texts: list[str]) -> list[str]:
        texts = self.en_de_model(texts)
        return self.de_en_model(texts)


class SGDDataset(Dataset):
    def __init__(self, dataset_dir: str, limit: int) -> None:
        self.aug_subdir = 'aug'
        self.dataset_dir = dataset_dir
        self.dialogs = self._load_dialogs()[:limit]
        self.augmented_dialogs = self._augment_dialogs()
        self.id_2_idx = self._create_mapping()

    def __getitem__(self, idx: int) -> Dialog:
        return self.dialogs[idx]

    def __len__(self) -> int:
        return len(self.dialogs)

    def compute_embeddings(
        self,
        cache_dir: str,
        model: Embedder,
        model_name: str,
    ) -> None:
        for idx in tqdm(range(len(self.dialogs)), desc='Computing embeddings...'):
            self.dialogs[idx].compute_embeddings(
                cache_dir, model, model_name,
            )
            self.augmented_dialogs[idx].compute_embeddings(
                cache_dir, model, model_name,
            )

    def load_clusters(self, clustering_filepath: str) -> None:
        with open(clustering_filepath) as clustering_file:
            clustering = json.load(clustering_file)
        for dialog_id, clusters in clustering.items():
            dialog_idx = self.id_2_idx[dialog_id]
            for idx in range(len(self.dialogs[dialog_idx].turns)):
                self.dialogs[dialog_idx].turns[idx].cluster = clusters[idx]

    def _create_mapping(self) -> dict[str, int]:
        id_2_idx = {}
        for idx, dialog in enumerate(self.dialogs):
            id_2_idx[dialog.dialog_id] = idx
        return id_2_idx

    def _load_dialogs(self) -> list[Dialog]:
        sub_dirs = ['train']

        dialogs_filepaths = []
        for sub_dir in sub_dirs:
            cur_dir = os.path.join(self.dataset_dir, sub_dir)
            dialogs_filenames = os.listdir(cur_dir)
            dialogs_filenames.remove('schema.json')
            dialogs_filepaths.extend(
                [os.path.join(cur_dir, filename) for filename in dialogs_filenames],
            )

        dialogs = []
        for dialogs_filepath in dialogs_filepaths:
            with open(dialogs_filepath) as dialogs_file:
                dialogs_data = json.load(dialogs_file)
                for dialog_dict in dialogs_data:
                    dialog = dialog_from_dict(dialog_dict)
                    dialogs.append(dialog)
        return dialogs

    def _augment_dialogs(self) -> list[Dialog]:
        augmented_dialogs = []
        os.makedirs(
            os.path.join(self.dataset_dir, self.aug_subdir),
            exist_ok=True,
        )

        model = AugmentationModel('cuda')

        for dialog in tqdm(self.dialogs, desc='Augmenting...'):
            augmented_dialogs.append(
                self._augment_dialog(deepcopy(dialog), model),
            )

        del model

        return augmented_dialogs

    def _augment_dialog(
        self,
        dialog: Dialog,
        model: AugmentationModel,
    ) -> Dialog:
        dialog.dialog_id = '{0}_aug'.format(dialog.dialog_id)
        dialog_filepath = os.path.join(
            self.dataset_dir,
            self.aug_subdir,
            '{0}.json'.format(dialog.dialog_id),
        )

        if os.path.isfile(dialog_filepath):
            with open(dialog_filepath) as dialog_file:
                utterances = json.load(dialog_file)
        else:
            utterances = model([turn.utterance for turn in dialog.turns])
            with open(dialog_filepath, 'w') as dialog_file:
                json.dump(utterances, dialog_file)

        for idx in range(len(utterances)):
            dialog.turns[idx].utterance = utterances[idx]

        return dialog
