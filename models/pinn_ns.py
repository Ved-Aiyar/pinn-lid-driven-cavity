import torch
import torch.nn as nn


class PINN_NS2D(nn.Module):
    """Physics-Informed Neural Network for 2D steady incompressible
    Navier-Stokes in a lid-driven cavity. Inputs: (x, y), Outputs: (u, v, p)."""

    def __init__(self, in_dim=2, out_dim=3, hidden_dim=64, n_hidden_layers=8):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: tensor of shape (N, 2) with columns [x, y].
        Returns: tensor (N, 3) with columns [u, v, p]."""
        return self.net(x)