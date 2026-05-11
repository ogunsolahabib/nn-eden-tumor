import torch
import torch.nn as nn
from rbf_layer import RBFLayer


class TumorClassifier(nn.Module):
    """
    Flexible RBF classifier.

    hidden_dims controls the post-RBF stack:
      ()        → RBF → output            (classic shallow RBF network)
      (32,)     → RBF → Dense32 → output  (1 hidden layer)
      (32, 16)  → RBF → Dense32 → Dense16 → output  (default, 2 hidden layers)

    BN + Dropout are applied only after the first hidden layer.
    All hidden activations use ReLU.
    """

    def __init__(
        self,
        n_centers:   int         = 20,
        hidden_dims: tuple       = (32, 16),
        dropout:     float       = 0.3,
    ):
        super().__init__()
        self.rbf = RBFLayer(in_features=2, n_centers=n_centers)

        layers = []
        in_dim = n_centers
        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            if i == 0:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if i == 0 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.rbf(x))

    def freeze_centers(self):
        self.rbf.centers.requires_grad_(False)

    def unfreeze_centers(self):
        self.rbf.centers.requires_grad_(True)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
