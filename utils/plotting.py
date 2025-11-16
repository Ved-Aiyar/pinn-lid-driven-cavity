import torch
import matplotlib.pyplot as plt
from config import DEVICE


def plot_velocity_field(model, nx: int = 41, ny: int = 41):
    """Quick quiver plot of (u, v) on a uniform grid in [0, 1] x [0, 1]."""
    model.eval()

    x = torch.linspace(0.0, 1.0, nx)
    y = torch.linspace(0.0, 1.0, ny)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    xy = torch.stack([X.flatten(), Y.flatten()], dim=1).to(DEVICE)

    with torch.no_grad():
        out = model(xy)
        u = out[:, 0].cpu().reshape(nx, ny)
        v = out[:, 1].cpu().reshape(nx, ny)

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    plt.figure()
    plt.quiver(X_np, Y_np, u, v)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Velocity field (u, v)")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.show()
