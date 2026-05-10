import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class RBFLayer(nn.Module):
    """
    First hidden layer: RBF activations with learnable centers and widths.
    Centers are seeded from k-means on the training data (unsupervised).
    Each unit computes: phi_i(x) = exp(-gamma_i * ||x - c_i||^2)
    """

    def __init__(self, in_features: int, n_centers: int):
        super().__init__()
        self.centers = nn.Parameter(torch.empty(n_centers, in_features))
        # log-gamma keeps gamma strictly positive during training
        self.log_gamma = nn.Parameter(torch.zeros(n_centers))
        nn.init.normal_(self.centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D)  ->  out: (B, K)
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)   # (B, K, D)
        dist_sq = (diff ** 2).sum(-1)                        # (B, K)
        gamma = torch.exp(self.log_gamma)                    # (K,)
        return torch.exp(-gamma * dist_sq)


def init_centers_from_kmeans(
    rbf_layer: RBFLayer,
    X_train: np.ndarray,
    seed: int = 42,
) -> KMeans:
    """
    Run k-means on raw (unlabeled) training features and copy cluster
    centroids into the RBF layer.  Width (gamma) is initialised from the
    mean nearest-neighbor distance between centroids so each basis function
    covers roughly one cluster cell.
    Returns the fitted KMeans object for inspection / plotting.
    """
    K = rbf_layer.centers.shape[0]
    km = KMeans(n_clusters=K, random_state=seed, n_init=15)
    km.fit(X_train)

    centers = km.cluster_centers_

    # Gamma from mean nearest-centroid distance
    D = pairwise_distances(centers)
    np.fill_diagonal(D, np.inf)
    sigma = D.min(axis=1).mean()
    gamma_init = 1.0 / (2.0 * sigma ** 2)

    with torch.no_grad():
        rbf_layer.centers.copy_(torch.tensor(centers, dtype=torch.float32))
        rbf_layer.log_gamma.fill_(float(np.log(gamma_init)))

    return km
