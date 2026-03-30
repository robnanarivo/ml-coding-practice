import torch as t
import torch.nn as nn
import pytest
from linear_regression import LinearRegression, LinearRegressionTrainer, MSEloss
from torch.utils.data import DataLoader, TensorDataset


class TestLinearRegressionInit:
    def test_weight_shape(self):
        model = LinearRegression(3, 5)
        assert model.weight.shape == (5, 3)

    def test_no_bias_by_default(self):
        model = LinearRegression(3, 5)
        assert model.bias is None

    def test_bias_shape(self):
        model = LinearRegression(3, 5, bias=True)
        assert model.bias is not None
        assert model.bias.shape == (5,)

    def test_is_nn_module(self):
        model = LinearRegression(3, 5)
        assert isinstance(model, nn.Module)

    def test_parameters_registered(self):
        model = LinearRegression(3, 5, bias=True)
        param_names = [name for name, _ in model.named_parameters()]
        assert "weight" in param_names
        assert "bias" in param_names


class TestLinearRegressionForward:
    def test_output_shape(self):
        model = LinearRegression(4, 2)
        x = t.randn(8, 4)
        out = model.forward(x)
        assert out.shape == (8, 2)

    def test_output_shape_with_bias(self):
        model = LinearRegression(4, 2, bias=True)
        x = t.randn(8, 4)
        out = model.forward(x)
        assert out.shape == (8, 2)

    def test_single_sample(self):
        model = LinearRegression(3, 1)
        x = t.randn(1, 3)
        out = model.forward(x)
        assert out.shape == (1, 1)

    def test_manual_computation(self):
        """Verify forward matches x @ W^T."""
        model = LinearRegression(3, 2)
        x = t.randn(4, 3)
        expected = x @ model.weight.T
        actual = model.forward(x)
        assert t.allclose(actual, expected)

    def test_manual_computation_with_bias(self):
        """Verify forward matches x @ W^T + b."""
        model = LinearRegression(3, 2, bias=True)
        x = t.randn(4, 3)
        expected = x @ model.weight.T + model.bias
        actual = model.forward(x)
        assert t.allclose(actual, expected)

    def test_gradients_flow(self):
        model = LinearRegression(3, 2)
        x = t.randn(4, 3)
        out = model.forward(x)
        loss = out.sum()
        loss.backward()
        assert model.weight.grad is not None


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


class TestMSEloss:
    def test_zero_loss(self):
        y = t.tensor([1.0, 2.0, 3.0])
        assert MSEloss(y, y).item() == 0.0

    def test_known_value(self):
        y_pred = t.tensor([1.0, 2.0])
        y = t.tensor([0.0, 0.0])
        # MSE = (1^2 + 2^2) / 2 = 2.5
        assert t.isclose(MSEloss(y_pred, y), t.tensor(2.5))

    def test_gradient_flows(self):
        y_pred = t.tensor([1.0, 2.0], requires_grad=True)
        y = t.tensor([0.0, 0.0])
        loss = MSEloss(y_pred, y)
        loss.backward()
        assert y_pred.grad is not None


class TestLinearRegressionTrainer:
    def test_train_returns_loss(self):
        model = LinearRegression(3, 1)
        train_loader, test_loader = make_loaders()
        trainer = LinearRegressionTrainer(model, epochs=1, train_loader=train_loader, test_loader=test_loader)
        loss = trainer.train()
        assert isinstance(loss, t.Tensor)
        assert loss.dim() == 0  # scalar

    def test_train_reduces_loss(self):
        """Evaluate loss should decrease after training."""
        t.manual_seed(0)
        model = LinearRegression(3, 1)
        train_loader, test_loader = make_loaders()
        trainer = LinearRegressionTrainer(model, epochs=0, train_loader=train_loader, test_loader=test_loader)
        initial_eval = trainer.evaluate()

        trainer.epochs = 100
        trainer.train()
        final_eval = trainer.evaluate()
        assert final_eval < initial_eval

    def test_evaluate_returns_scalar(self):
        model = LinearRegression(3, 1)
        train_loader, test_loader = make_loaders()
        trainer = LinearRegressionTrainer(model, epochs=1, train_loader=train_loader, test_loader=test_loader)
        trainer.train()
        error = trainer.evaluate()
        assert error >= 0

    def test_weights_update_after_training(self):
        model = LinearRegression(3, 1)
        train_loader, test_loader = make_loaders()
        initial_weight = model.weight.clone()
        trainer = LinearRegressionTrainer(model, epochs=5, train_loader=train_loader, test_loader=test_loader)
        trainer.train()
        assert not t.equal(model.weight, initial_weight)
