import torch
from torch import nn
from torch.optim import Optimizer
from pytorch_metric_learning.losses import BaseMetricLossFunction


class GRUEmbedder(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rnn(x)[0][:, -1, :]


def train_step(
    model: GRUEmbedder,
    batch: torch.Tensor,
    criterion: BaseMetricLossFunction,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    optimizer.zero_grad()

    batch_size = batch.size(0)
    # [0, 1, ... n - 1, 0, 1 ... n - 1]
    labels = torch.LongTensor(list(range(batch_size // 2)) * 2)
    batch, labels = batch.to(device), labels.to(device)
    embeddings = model(batch)

    loss = criterion(embeddings, labels)
    loss.backward()
    optimizer.step()

    return loss.item()
