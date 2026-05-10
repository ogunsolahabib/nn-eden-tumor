import torch
import torch.nn as nn
from rbf_layer import RBFLayer


class TumorClassifier(nn.Module):
    """
    Architecture
    ------------
    Layer 1  — RBFLayer (k centers, k-means seeded)
        Encodes spatial proximity to k cluster centroids.
        Transforms the non-convex ring geometry into a representable
        metric space where class membership correlates with proximity patterns.

    Layer 2  — Linear(k -> 32) + BatchNorm + ReLU
        Learns which linear combinations of RBF responses delimit the viable
        rim.  A single linear layer on top of RBFs (classic RBF network) cannot
        express the compound "inside outer boundary AND outside inner core"
        predicate without a very large k; a nonlinear layer compensates with
        far fewer parameters.

    Layer 3  — Linear(32 -> 16) + ReLU
        Captures the directional (hot/cold) asymmetry of the C2 immune ring,
        which is not radially symmetric and therefore not fully resolved by the
        radially symmetric RBF activations alone.

    Output   — Linear(16 -> 1)   (raw logit, sigmoid inside BCEWithLogitsLoss)
    """

    def __init__(self, n_centers: int = 20, dropout: float = 0.3):
        super().__init__()
        self.rbf     = RBFLayer(in_features=2, n_centers=n_centers)
        self.hidden1 = nn.Linear(n_centers, 32)
        self.bn1     = nn.BatchNorm1d(32)
        self.hidden2 = nn.Linear(32, 16)
        self.out     = nn.Linear(16, 1)
        self.drop    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rbf(x)
        x = torch.relu(self.bn1(self.hidden1(x)))
        x = self.drop(x)
        x = torch.relu(self.hidden2(x))
        return self.out(x)   # raw logit

    def freeze_centers(self):
        self.rbf.centers.requires_grad_(False)

    def unfreeze_centers(self):
        self.rbf.centers.requires_grad_(True)
