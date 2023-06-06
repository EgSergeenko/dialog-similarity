import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
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

    # [0, 1, ... n - 1, 0, 1 ... n - 1]
    labels = torch.LongTensor(list(range(batch.size(0) // 2)) * 2)
    batch, labels = batch.to(device), labels.to(device)
    embeddings = model(batch)

    loss = criterion(embeddings, labels)
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def eval_step(
    model: GRUEmbedder,
    batch: torch.Tensor,
    criterion: BaseMetricLossFunction,
    device: torch.device,
) -> float:
    labels = torch.LongTensor(list(range(batch.size(0) // 2)) * 2)
    batch, labels = batch.to(device), labels.to(device)
    embeddings = model(batch)

    loss = criterion(embeddings, labels)

    return loss.item()


def train_epoch(
    model: GRUEmbedder,
    dataloader: DataLoader,
    criterion: BaseMetricLossFunction,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0
    for batch in dataloader:
        step_loss = train_step(
            model=model,
            batch=batch,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        total_loss += step_loss
    return total_loss / len(dataloader)


def eval_epoch(
    model: GRUEmbedder,
    dataloader: DataLoader,
    criterion: BaseMetricLossFunction,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0
    for batch in dataloader:
        step_loss = eval_step(
            model=model,
            batch=batch,
            criterion=criterion,
            device=device,
        )
        total_loss += step_loss
    return total_loss / len(dataloader)

