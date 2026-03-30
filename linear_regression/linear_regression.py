import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from jaxtyping import Float
from torch import Tensor

class LinearRegression(nn.Module):
    def __init__(self, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(t.randn(d_out, d_in))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(t.zeros(d_out))
    def forward(
            self, 
            x: Float[Tensor, "batch d_in"]
        ) -> Float[Tensor, "batch d_out"]:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

def MSEloss(y_pred, y):
    residual = y_pred - y
    MSE = residual.square().mean()
    return MSE

class LinearRegressionTrainer:
    def __init__(self, model: LinearRegression, epochs: int, train_loader: DataLoader, test_loader: DataLoader):
        self.model = model
        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=1e-2
        )
        self.epochs = epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
    
    def train(self) -> Float[Tensor, ""]:
        for epoch in range(self.epochs):
            epoch_loss = []
            for i, (X, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                y_pred = self.model(X)
                loss = MSEloss(y_pred, y)
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
            print(f"Train loss for epoch {epoch}: {np.mean(epoch_loss)}")
        return loss

    @t.inference_mode()
    def evaluate(self):
        error_list = []
        for X, y in self.test_loader:
            y_pred = self.model(X)
            error = MSEloss(y_pred, y)
            error_list.append(error.item())
        mean_error = np.mean(error_list)
        return mean_error

def make_loaders(n_train=64, n_test=16, d_in=3, d_out=1, batch_size=16, noise_std=0.1):
    """Create train/test loaders with a linear relationship plus Gaussian noise."""
    t.manual_seed(42)
    w_true = t.randn(d_out, d_in)

    X_train = t.randn(n_train, d_in)
    y_train = X_train @ w_true.T + noise_std * t.randn(n_train, d_out)
    X_test = t.randn(n_test, d_in)
    y_test = X_test @ w_true.T + noise_std * t.randn(n_test, d_out)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    return train_loader, test_loader

if __name__ == "__main__":
    t.manual_seed(0)
    model = LinearRegression(3, 1)
    train_loader, test_loader = make_loaders()
    trainer = LinearRegressionTrainer(model, epochs=0, train_loader=train_loader, test_loader=test_loader)
    initial_eval = trainer.evaluate()

    trainer.epochs = 100
    trainer.train()
    final_eval = trainer.evaluate()
    print(f"initial loss: {initial_eval}\nfinal loss: {final_eval}")