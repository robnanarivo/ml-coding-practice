import torch as t
import torch.nn as nn
from jaxtyping import Float
from dataclasses import dataclass, field
from torch.utils.data import DataLoader


class LogisticRegression(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(
            self, 
            x: Float[Tensor, "batch n_features"]
        ) -> Float[Tensor, "batch n_classes"]:
        """Output probs"""
        logits = self.linear(x)
        out = self.softmax(logits)
        return out

    def softmax(self, x: Float[Tensor, "batch n_features"]) -> Float[Tensor, "batch n_features"]:
        prob_sum = x.exp().sum(dim=-1)
        probs = x.exp() / prob_sum
        return probs

@dataclass
class TrainConfig:
    lr: float = 1e-4
    epochs: int = 3
    train_loader: DataLoader
    test_loader: DataLoader


class LogisticRegressionTrainer:
    def __init__(self, model: LogisticRegression, config: TrainConfig):
        self.model = model
        self.epochs = config.epochs
        self.optimizer = t.optim.AdamW(config.lr)
        self.train_loader = config.train_loader
        self.test_loader = config.test_loader

    def train(self) -> Float[Tensor, ""]:
        for epoch in range(self.epochs):
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(X)
                loss = self.cross_entropy(y, y_pred)
                loss.backward()
                self.optimizer.step()
        return loss

    @t.inference_mode()
    def evaluate(self):
        for X, y in self.test_loader:


    def cross_entropy(
            self, 
            y: Float[Tensor, "batch n_classes"], 
            y_pred: Float[Tensor, "batch n_classes"]
        ) -> Float[Tensor, ""]:
        ce = -(y * y_pred.log()).sum(dim=-1).mean()
        return ce