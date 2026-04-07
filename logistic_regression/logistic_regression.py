import torch as t
import torch.nn as nn
from jaxtyping import Float
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch import Tensor
import numpy as np


class LogisticRegression(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)
        self.n_classes = n_classes
        self.n_features = n_features

    def forward(
            self, 
            x: Float[Tensor, "batch n_features"]
        ) -> Float[Tensor, "batch n_classes"]:
        """Output probs"""
        logits = self.linear(x)
        out = self.softmax(logits)
        return out

    def softmax(self, x: Float[Tensor, "batch n_classes"]) -> Float[Tensor, "batch n_classes"]:
        prob_sum = x.exp().sum(dim=-1)
        # print(f"shape: x: {x.shape}, prob_sum: {prob_sum.shape}")
        probs = x.exp() / prob_sum.unsqueeze(-1)
        return probs

@dataclass
class TrainConfig:
    train_loader: DataLoader
    test_loader: DataLoader
    lr: float = 1e-4
    epochs: int = 3


class LogisticRegressionTrainer:
    def __init__(self, model: LogisticRegression, config: TrainConfig):
        self.model = model
        self.epochs = config.epochs
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.train_loader = config.train_loader
        self.test_loader = config.test_loader

    def train(self) -> list[float]:
        """assume y is one-hot encoded"""
        loss_list = []
        for epoch in range(self.epochs):
            epoch_loss_list = []
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                probs = self.model(X)
                loss = self.cross_entropy(y, probs)
                epoch_loss_list.append(loss.item())
                loss.backward()
                self.optimizer.step()
            epoch_loss = np.mean(epoch_loss_list)
            loss_list.append(epoch_loss)
            print(f"epoch loss for epoch {epoch}: {epoch_loss}")
        return loss_list

    @t.inference_mode()
    def evaluate(self) -> tuple[Float[Tensor, "n_classes"], Float[Tensor, "n_classes"]]:
        n_classes = self.model.n_classes
        true_pos = t.zeros(n_classes)
        false_pos = t.zeros(n_classes)
        false_neg = t.zeros(n_classes)
        for X, y in self.test_loader:
            probs = self.model(X)
            y_label = y.argmax(dim=-1)
            y_pred_label = probs.argmax(dim=-1)
            for c in range(n_classes):
                true_pos[c] += ((y_pred_label == c) & (y_label == c)).float().sum()
                false_pos[c] += ((y_pred_label == c) & (y_label != c)).float().sum()
                false_neg[c] += ((y_pred_label != c) & (y_label == c)).float().sum()

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return precision, recall

    def cross_entropy(
            self, 
            y: Float[Tensor, "batch n_classes"], 
            probs: Float[Tensor, "batch n_classes"]
        ) -> Float[Tensor, ""]:
        ce = -(y * probs.log()).sum(dim=-1).mean()
        return ce