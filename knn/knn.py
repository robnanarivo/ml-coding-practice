import torch as t
from jaxtyping import Float
from torch import Tensor
from collections.abc import Callable
import einops

class KNN:
    def __init__(self, 
            k: int, 
            X_train: Float[Tensor, "sample n_features"], 
            y_train: Float[Tensor, "sample"], 
            distance_func: Callable[[Float[Tensor, "n_features"], Float[Tensor, "n_features"]], Float]
        ):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train
        self.distance_func = distance_func
    
    def predict(self, X: Float[Tensor, "batch n_features"]) -> Float[Tensor, "batch"]:
        n_points = self.X_train.size(0)
        batch_size = X.size(0)
        X = einops.repeat(X, "batch n_features -> batch n_points n_features", n_points=n_points)
        X_train = einops.repeat(self.X_train, "n_points n_features -> batch n_points n_features", batch=batch_size)
        batched_distance_func = t.vmap(t.vmap(self.distance_func))
        distance = batched_distance_func(X, X_train)
        _, indices = distance.topk(k=self.k, largest=False) # indices: [batch, k]
        labels = self.y_train[indices]
        y_pred, _ = labels.mode(dim=-1)
        return y_pred

def euclidean_dist(p1: Float[Tensor, "coordinate"], p2: Float[Tensor, "coordinate"]) -> Float[Tensor, ""]:
    return t.norm(p1 - p2)