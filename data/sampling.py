import torch
from config import (
    DOMAIN_X_MIN, DOMAIN_X_MAX,
    DOMAIN_Y_MIN, DOMAIN_Y_MAX,
    U_LID,
    DEVICE,
)


def sample_interior(n_points: int):
    """Randomly sample interior points in the open domain (0,1)x(0,1).
    Returns x, y tensors of shape (N, 1)."""
    x = torch.rand(n_points, 1) * (DOMAIN_X_MAX - DOMAIN_X_MIN) + DOMAIN_X_MIN
    y = torch.rand(n_points, 1) * (DOMAIN_Y_MAX - DOMAIN_Y_MIN) + DOMAIN_Y_MIN

    # Avoid sampling exactly on boundaries
    eps = 1e-6
    x = x.clamp(DOMAIN_X_MIN + eps, DOMAIN_X_MAX - eps)
    y = y.clamp(DOMAIN_Y_MIN + eps, DOMAIN_Y_MAX - eps)

    return x.to(DEVICE), y.to(DEVICE)


def sample_boundary(n_points_total: int):
    """Sample boundary points roughly evenly on all four sides.
    Returns:
        x_b, y_b: (N_b, 1)
        u_b, v_b: (N_b, 1) target velocities at boundary
    """
    n_side = n_points_total // 4

    # Left wall: x=0, u=v=0
    y_left = torch.rand(n_side, 1) * (DOMAIN_Y_MAX - DOMAIN_Y_MIN) + DOMAIN_Y_MIN
    x_left = torch.zeros_like(y_left) + DOMAIN_X_MIN
    u_left = torch.zeros_like(x_left)
    v_left = torch.zeros_like(x_left)

    # Right wall: x=1, u=v=0
    y_right = torch.rand(n_side, 1) * (DOMAIN_Y_MAX - DOMAIN_Y_MIN) + DOMAIN_Y_MIN
    x_right = torch.zeros_like(y_right) + DOMAIN_X_MAX
    u_right = torch.zeros_like(x_right)
    v_right = torch.zeros_like(x_right)

    # Bottom wall: y=0, u=v=0
    x_bottom = torch.rand(n_side, 1) * (DOMAIN_X_MAX - DOMAIN_X_MIN) + DOMAIN_X_MIN
    y_bottom = torch.zeros_like(x_bottom) + DOMAIN_Y_MIN
    u_bottom = torch.zeros_like(x_bottom)
    v_bottom = torch.zeros_like(x_bottom)

    # Top lid: y=1, u=U_LID, v=0
    x_top = torch.rand(n_side, 1) * (DOMAIN_X_MAX - DOMAIN_X_MIN) + DOMAIN_X_MIN
    y_top = torch.zeros_like(x_top) + DOMAIN_Y_MAX
    u_top = torch.zeros_like(x_top) + U_LID
    v_top = torch.zeros_like(x_top)

    x_b = torch.cat([x_left, x_right, x_bottom, x_top], dim=0)
    y_b = torch.cat([y_left, y_right, y_bottom, y_top], dim=0)
    u_b = torch.cat([u_left, u_right, u_bottom, u_top], dim=0)
    v_b = torch.cat([v_left, v_right, v_bottom, v_top])

    return x_b.to(DEVICE), y_b.to(DEVICE), u_b.to(DEVICE), v_b.to(DEVICE)

