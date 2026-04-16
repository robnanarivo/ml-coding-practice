import torch as t
import pytest
from knn import KNN, euclidean_dist


def make_simple_data():
    """Create a simple 2D dataset with two classes (0 and 1)."""
    t.manual_seed(42)
    # Class 0: centered around (0, 0)
    X_class0 = t.randn(20, 2) * 0.5 + t.tensor([0.0, 0.0])
    # Class 1: centered around (5, 5)
    X_class1 = t.randn(20, 2) * 0.5 + t.tensor([5.0, 5.0])

    X_train = t.cat([X_class0, X_class1], dim=0)
    y_train = t.cat([t.zeros(20), t.ones(20)])
    return X_train, y_train


class TestEuclideanDist:
    def test_same_point(self):
        p = t.tensor([1.0, 2.0, 3.0])
        assert euclidean_dist(p, p).item() == 0.0

    def test_known_distance(self):
        p1 = t.tensor([0.0, 0.0])
        p2 = t.tensor([3.0, 4.0])
        assert t.isclose(euclidean_dist(p1, p2), t.tensor(5.0))

    def test_symmetry(self):
        p1 = t.tensor([1.0, 2.0])
        p2 = t.tensor([4.0, 6.0])
        assert t.isclose(euclidean_dist(p1, p2), euclidean_dist(p2, p1))

    def test_1d(self):
        p1 = t.tensor([3.0])
        p2 = t.tensor([7.0])
        assert t.isclose(euclidean_dist(p1, p2), t.tensor(4.0))


class TestKNNInit:
    def test_stores_attributes(self):
        X_train, y_train = make_simple_data()
        knn = KNN(k=3, X_train=X_train, y_train=y_train, distance_func=euclidean_dist)
        assert knn.k == 3
        assert t.equal(knn.X_train, X_train)
        assert t.equal(knn.y_train, y_train)
        assert knn.distance_func is euclidean_dist


class TestKNNPredict:
    def test_output_shape(self):
        X_train, y_train = make_simple_data()
        knn = KNN(k=3, X_train=X_train, y_train=y_train, distance_func=euclidean_dist)
        X_test = t.randn(5, 2)
        y_pred = knn.predict(X_test)
        assert y_pred.shape == (5,)

    def test_single_sample(self):
        X_train, y_train = make_simple_data()
        knn = KNN(k=3, X_train=X_train, y_train=y_train, distance_func=euclidean_dist)
        X_test = t.tensor([[0.0, 0.0]])
        y_pred = knn.predict(X_test)
        assert y_pred.shape == (1,)

    def test_predicts_nearby_class(self):
        """Points near class 0 center should be predicted as 0, and vice versa."""
        X_train, y_train = make_simple_data()
        knn = KNN(k=3, X_train=X_train, y_train=y_train, distance_func=euclidean_dist)
        X_test = t.tensor([
            [0.1, 0.1],   # near class 0
            [4.9, 5.1],   # near class 1
        ])
        y_pred = knn.predict(X_test)
        assert y_pred[0].item() == 0.0
        assert y_pred[1].item() == 1.0

    def test_k_equals_1(self):
        """With k=1, should return label of the nearest neighbor."""
        X_train = t.tensor([[0.0, 0.0], [10.0, 10.0]])
        y_train = t.tensor([0.0, 1.0])
        knn = KNN(k=1, X_train=X_train, y_train=y_train, distance_func=euclidean_dist)
        X_test = t.tensor([[0.1, 0.1]])
        y_pred = knn.predict(X_test)
        assert y_pred[0].item() == 0.0

    def test_batch_prediction(self):
        """All predictions in a batch should be correct for well-separated data."""
        X_train, y_train = make_simple_data()
        knn = KNN(k=5, X_train=X_train, y_train=y_train, distance_func=euclidean_dist)
        X_test = t.tensor([
            [0.0, 0.0],
            [0.2, -0.1],
            [5.0, 5.0],
            [5.2, 4.8],
        ])
        y_pred = knn.predict(X_test)
        expected = t.tensor([0.0, 0.0, 1.0, 1.0])
        assert t.equal(y_pred, expected)

    def test_custom_distance_func(self):
        """KNN should work with a custom distance function (Manhattan distance)."""
        def manhattan_dist(p1, p2):
            return t.sum(t.abs(p1 - p2))

        X_train = t.tensor([[0.0, 0.0], [10.0, 10.0]])
        y_train = t.tensor([0.0, 1.0])
        knn = KNN(k=1, X_train=X_train, y_train=y_train, distance_func=manhattan_dist)
        X_test = t.tensor([[1.0, 1.0]])
        y_pred = knn.predict(X_test)
        assert y_pred[0].item() == 0.0
