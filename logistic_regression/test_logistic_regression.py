import torch as t
import torch.nn as nn
import pytest
from logistic_regression import LogisticRegression, LogisticRegressionTrainer, TrainConfig
from torch.utils.data import DataLoader, TensorDataset


class TestLogisticRegressionInit:
    def test_linear_layer_shape(self):
        model = LogisticRegression(4, 3)
        assert model.linear.weight.shape == (3, 4)
        assert model.linear.bias.shape == (3,)

    def test_is_nn_module(self):
        model = LogisticRegression(4, 3)
        assert isinstance(model, nn.Module)

    def test_parameters_registered(self):
        model = LogisticRegression(4, 3)
        param_names = [name for name, _ in model.named_parameters()]
        assert "linear.weight" in param_names
        assert "linear.bias" in param_names


class TestSoftmax:
    def test_output_sums_to_one(self):
        model = LogisticRegression(3, 4)
        x = t.randn(8, 4)
        probs = model.softmax(x)
        sums = probs.sum(dim=-1)
        assert t.allclose(sums, t.ones(8), atol=1e-5)

    def test_output_non_negative(self):
        model = LogisticRegression(3, 4)
        x = t.randn(8, 4)
        probs = model.softmax(x)
        assert (probs >= 0).all()

    def test_output_shape(self):
        model = LogisticRegression(3, 4)
        x = t.randn(8, 4)
        probs = model.softmax(x)
        assert probs.shape == (8, 4)


class TestLogisticRegressionForward:
    def test_output_shape(self):
        model = LogisticRegression(4, 3)
        x = t.randn(8, 4)
        out = model.forward(x)
        assert out.shape == (8, 3)

    def test_output_is_probability(self):
        model = LogisticRegression(4, 3)
        x = t.randn(8, 4)
        out = model.forward(x)
        assert (out >= 0).all()
        assert t.allclose(out.sum(dim=-1), t.ones(8), atol=1e-5)

    def test_single_sample(self):
        model = LogisticRegression(3, 2)
        x = t.randn(1, 3)
        out = model.forward(x)
        assert out.shape == (1, 2)

    def test_gradients_flow(self):
        model = LogisticRegression(3, 2)
        x = t.randn(4, 3)
        out = model.forward(x)
        loss = out.sum()
        loss.backward()
        assert model.linear.weight.grad is not None


class TestCrossEntropy:
    def test_perfect_prediction(self):
        trainer = make_trainer(n_features=3, n_classes=2)
        y = t.tensor([[1.0, 0.0], [0.0, 1.0]])
        probs = t.tensor([[0.99, 0.01], [0.01, 0.99]])
        loss = trainer.cross_entropy(y, probs)
        assert loss.item() < 0.02

    def test_bad_prediction_high_loss(self):
        trainer = make_trainer(n_features=3, n_classes=2)
        y = t.tensor([[1.0, 0.0], [0.0, 1.0]])
        probs = t.tensor([[0.01, 0.99], [0.99, 0.01]])
        loss = trainer.cross_entropy(y, probs)
        assert loss.item() > 4.0

    def test_loss_is_non_negative(self):
        trainer = make_trainer(n_features=3, n_classes=3)
        y = t.tensor([[1.0, 0.0, 0.0]])
        probs = t.tensor([[0.7, 0.2, 0.1]])
        loss = trainer.cross_entropy(y, probs)
        assert loss.item() >= 0

    def test_gradient_flows(self):
        trainer = make_trainer(n_features=3, n_classes=2)
        probs = t.tensor([[0.7, 0.3], [0.4, 0.6]], requires_grad=True)
        y = t.tensor([[1.0, 0.0], [0.0, 1.0]])
        loss = trainer.cross_entropy(y, probs)
        loss.backward()
        assert probs.grad is not None


def make_loaders(n_train=128, n_test=32, n_features=3, n_classes=3, batch_size=16):
    """Create train/test loaders with linearly separable classes."""
    t.manual_seed(42)
    w_true = t.randn(n_classes, n_features)

    X_train = t.randn(n_train, n_features)
    logits_train = X_train @ w_true.T
    y_train = t.zeros(n_train, n_classes)
    y_train.scatter_(1, logits_train.argmax(dim=-1, keepdim=True), 1.0)

    X_test = t.randn(n_test, n_features)
    logits_test = X_test @ w_true.T
    y_test = t.zeros(n_test, n_classes)
    y_test.scatter_(1, logits_test.argmax(dim=-1, keepdim=True), 1.0)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    return train_loader, test_loader


def make_trainer(n_features=3, n_classes=3, epochs=1, lr=1e-2):
    """Helper to create a trainer with default loaders."""
    train_loader, test_loader = make_loaders(n_features=n_features, n_classes=n_classes)
    model = LogisticRegression(n_features, n_classes)
    config = TrainConfig(train_loader=train_loader, test_loader=test_loader, lr=lr, epochs=epochs)
    return LogisticRegressionTrainer(model, config)


class TestLogisticRegressionTrainer:
    def test_train_returns_loss_list(self):
        trainer = make_trainer(epochs=2)
        loss_list = trainer.train()
        assert isinstance(loss_list, list)
        assert len(loss_list) == 2
        assert isinstance(loss_list[0], float)

    def test_train_reduces_loss(self):
        t.manual_seed(0)
        trainer = make_trainer(epochs=100, lr=1e-2)
        loss_list = trainer.train()
        assert loss_list[-1] < loss_list[0]

    def test_evaluate_returns_precision_recall(self):
        trainer = make_trainer(epochs=1)
        trainer.train()
        precision, recall = trainer.evaluate()
        assert precision.shape == (3,)
        assert recall.shape == (3,)
        assert (precision >= 0).all() and (precision <= 1).all()
        assert (recall >= 0).all() and (recall <= 1).all()

    def test_weights_update_after_training(self):
        model = LogisticRegression(3, 3)
        train_loader, test_loader = make_loaders()
        initial_weight = model.linear.weight.clone()
        config = TrainConfig(train_loader=train_loader, test_loader=test_loader, lr=1e-2, epochs=5)
        trainer = LogisticRegressionTrainer(model, config)
        trainer.train()
        assert not t.equal(model.linear.weight, initial_weight)
