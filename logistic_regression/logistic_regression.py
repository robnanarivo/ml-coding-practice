import torch as t
import torch.nn as nn
from jaxtyping import Float


class LogisticRegression(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes),

    def forward(
            self, 
            x: Float[Tensor, "batch n_features"]
        ) -> Float[Tensor, "batch n_classes"]:
        """Output probs"""
        logits = self.linear(x)
        out = self.softmax(logits)
        return out

    def softmax(self, x):
        prob_sum = x.exp().sum(dim=-1)
        probs = x.exp() / prob_sum
        return probs

class LogisticRegressionTrainer:
    def __init__(self):
        pass

    def train(self):
        pass

    @t.inference_mode()
    def evaluate(self):
        pass