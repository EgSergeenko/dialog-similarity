import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizer
from transformers.models.bert import BertModel
from transformers.models.roberta import RobertaModel


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

        if device != model.device.type:
            model.to(device)

        self.model.eval()

    @torch.no_grad()
    def __call__(self, text: str) -> np.ndarray:
        if self._is_bert():
            return self._inference_bert(text)
        return self._inference_sentence_transformer(text)

    def _inference_bert(self, text: str) -> np.ndarray:
        input_ids = self.tokenizer(
            text, return_tensors='pt',
        )['input_ids'].to(self.device)

        embeddings = self.model(input_ids)
        return torch.mean(embeddings[0], dim=-2).squeeze().cpu().numpy()

    def _inference_sentence_transformer(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]

    def _is_bert(self) -> bool:
        return isinstance(self.model, BertModel) or isinstance(self.model, RobertaModel)