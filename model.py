import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer
from transformers.models.bert import BertModel
from transformers.models.roberta import RobertaModel


class ClusterEmbedder(object):
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.clustering, self.embedding_mapping = self._load_model()

    def _load_model(self) -> (dict[str, int], dict[int, np.ndarray]):
        clustering_filepath = os.path.join(self.model_path, 'clustering.json')
        with open(clustering_filepath) as clustering_file:
            clustering = json.load(clustering_file)

        embeddings_dir = os.path.join(self.model_path, 'embeddings')
        embedding_filenames = os.listdir(embeddings_dir)
        embedding_mapping = {}
        for embedding_filename in embedding_filenames:
            cluster_id = int(embedding_filename.split('.')[0])
            embedding_filepath = os.path.join(
                embeddings_dir, embedding_filename,
            )
            embedding_mapping[cluster_id] = np.load(embedding_filepath)

        return clustering, embedding_mapping

    def __call__(self, text: str) -> np.ndarray:
        if text not in self.clustering:
            return np.zeros(2)
        cluster_id = self.clustering[text]
        return self.embedding_mapping[cluster_id]


class Embedder(object):
    def __init__(
        self,
        model: SentenceTransformer | BertModel | RobertaModel,
        tokenizer: PreTrainedTokenizer | None,
        device: str = 'cpu',
    ) -> None:

        self.model = model
        if self._is_bert() and tokenizer is None:
            raise ValueError('For this model class fill tokenizer param.')
        self.tokenizer = tokenizer
        self.device = device

        if not isinstance(self.model, ClusterEmbedder):
            if device != model.device.type:
                model.to(device)
            self.model.eval()

    @torch.no_grad()
    def __call__(self, text: str) -> np.ndarray:
        if self._is_bert():
            return self._inference_bert(text)
        elif isinstance(self.model, ClusterEmbedder):
            return self._inference_cluster_embedding(text)
        return self._inference_sentence_transformer(text)

    def _inference_bert(self, text: str) -> np.ndarray:
        input_ids = self.tokenizer(
            text, return_tensors='pt',
        )['input_ids'].to(self.device)

        embeddings = self.model(input_ids)
        return torch.mean(embeddings[0], dim=-2).squeeze().cpu().numpy()

    def _inference_cluster_embedding(self, text: str) -> np.ndarray:
        return self.model(text)

    def _inference_sentence_transformer(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]

    def _is_bert(self) -> bool:
        return isinstance(self.model, BertModel) or isinstance(self.model, RobertaModel)
