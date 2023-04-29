import json
import os

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from dialog import dialog_from_dict, Dialog


class SGDDataset(object):
    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = dataset_dir
        self.dialogs = self._preprocess_dataset()

    def __getitem__(self, idx: int) -> Dialog:
        return self.dialogs[idx]

    def __len__(self) -> int:
        return len(self.dialogs)

    def compute_embeddings(
        self,
        cache_dir: str,
        model: SentenceTransformer,
        model_name: str,
    ) -> None:
        for dialog in tqdm(self.dialogs):
            dialog.compute_embeddings(cache_dir, model, model_name)

    def _preprocess_dataset(self) -> list[Dialog]:
        sub_dirs = ['dev', 'test', 'train']

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
