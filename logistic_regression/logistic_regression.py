import torch as t
import torch.nn as nn
from jaxtyping import Float
from dataclasses import dataclass, field
from torch.utils.data import DataLoader


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
        self.optimizer = t.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.train_loader = config.train_loader
        self.test_loader = config.test_loader

    def train(self) -> Float[Tensor, ""]:
        """assume y is one-hot encoded"""
        for epoch in range(self.epochs):
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                probs = self.model(X)
                loss = self.cross_entropy(y, probs)
                loss.backward()
                self.optimizer.step()
        return loss

    @t.inference_mode()
    def evaluate(self) -> tuple[Float[Tensor, "n_classes"], Float[Tensor, "n_classes"]]:
        n_classes = self.model.n_classes
        true_pos = t.zeros(n_classes)
        false_pos = t.zeros(n_classes)
        false_neg = t.zeros(n_classes)
        total = 0
        for X, y in self.test_loader:
            probs = self.model(X)
            y_label = y.argmax(dim=-1)
            y_pred_label = probs.argmax(dim=-1)
            total += len(y_label)
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