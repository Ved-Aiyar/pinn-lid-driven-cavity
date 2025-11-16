import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import DEVICE, INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, N_HIDDEN_LAYERS
from models.pinn_ns import PINN_NS2D
from utils.plotting import plot_velocity_field

# Paths
HERE = os.path.dirname(__file__)
CHECKPOINT_DIR = os.path.join(HERE, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "pinn_ns_cavity.pth")
HISTORY_PATH = os.path.join(CHECKPOINT_DIR, "training_history.pth")
CFD_RESULTS_PATH = os.path.join(HERE, "cfd", "results", "cavity_Re100.npz")


# -------------------------
#  Model loading utilities
# -------------------------

def load_trained_model():
    """Load the trained PINN from checkpoint."""
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found at {CHECKPOINT_PATH}. "
            "Run main_train.py once to train and save the model."
        )

    model = PINN_NS2D(
        in_dim=INPUT_DIM,
        out_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_hidden_layers=N_HIDDEN_LAYERS,
    ).to(DEVICE)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded trained model from {CHECKPOINT_PATH}")
    return model


# -------------------------
#  Load CFD results
# -------------------------

def load_cfd_results():
    """Load CFD lid-driven cavity results from NPZ file."""
    if not os.path.exists(CFD_RESULTS_PATH):
        raise FileNotFoundError(
            f"CFD results not found at {CFD_RESULTS_PATH}. "
            "Run cfd/cfd_solver.py first."
        )

    data = np.load(CFD_RESULTS_PATH)
    x = data["x"]          # (nx,)
    y = data["y"]          # (ny,)
    u = data["u"]          # (nx, ny)
    v = data["v"]          # (nx, ny)
    p = data["p"]          # (nx, ny)
    print(f"Loaded CFD results from {CFD_RESULTS_PATH}")
    return x, y, u, v, p


# -------------------------
#  Evaluate PINN on grid
# -------------------------

def evaluate_pinn_on_grid(model, x: np.ndarray, y: np.ndarray):
    """
    Evaluate PINN (u, v, p) on given 1D grid arrays x, y.

    x: (nx,) numpy
    y: (ny,) numpy

    Returns:
        U: (nx, ny) numpy array of u(x,y)
        V: (nx, ny) numpy array of v(x,y)
        P: (nx, ny) numpy array of p(x,y)
    """
    nx = x.shape[0]
    ny = y.shape[0]

    # Convert to torch
    x_t = torch.from_numpy(x.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    X_t, Y_t = torch.meshgrid(x_t, y_t, indexing="ij")

    xy = torch.stack([X_t.flatten(), Y_t.flatten()], dim=1).to(DEVICE)

    with torch.no_grad():
        out = model(xy)
        u = out[:, 0].cpu().reshape(nx, ny).numpy()
        v = out[:, 1].cpu().reshape(nx, ny).numpy()
        p = out[:, 2].cpu().reshape(nx, ny).numpy()

    return u, v, p


# -------------------------
#  Training history plot
# -------------------------

def plot_training_history():
    """Plot loss vs epoch from saved training_history.pth."""
    if not os.path.exists(HISTORY_PATH):
        print(f"No training history found at {HISTORY_PATH}")
        return

    history = torch.load(HISTORY_PATH, map_location="cpu")

    epochs = np.array(history["epoch"])
    total = np.array(history["total"])
    pde = np.array(history["pde"])
    bc = np.array(history["bc"])

    plt.figure()
    plt.semilogy(epochs, total, label="Total loss")
    plt.semilogy(epochs, pde, label="PDE loss")
    plt.semilogy(epochs, bc, label="BC loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("PINN training loss vs epoch")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()


# -------------------------
#  Vorticity computation
# -------------------------

def compute_vorticity(x: np.ndarray, y: np.ndarray,
                      U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute vorticity omega = dv/dx - du/dy using central finite differences.
    x, y: 1D arrays
    U, V: 2D arrays (nx, ny)
    """
    nx, ny = U.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    omega = np.zeros_like(U)

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            dv_dx = (V[i + 1, j] - V[i - 1, j]) / (2.0 * dx)
            du_dy = (U[i, j + 1] - U[i, j - 1]) / (2.0 * dy)
            omega[i, j] = dv_dx - du_dy

    # Simple boundary fill
    omega[0, :] = omega[1, :]
    omega[-1, :] = omega[-2, :]
    omega[:, 0] = omega[:, 1]
    omega[:, -1] = omega[:, -2]

    return omega


# -------------------------
#  Centerline comparison
# -------------------------

def plot_centerline_comparison(x, y, u_cfd, v_cfd, U_pinn, V_pinn):
    """Compare u(y) at x=0.5 and v(x) at y=0.5 for CFD vs PINN."""
    x = np.asarray(x)
    y = np.asarray(y)

    ix_mid = np.argmin(np.abs(x - 0.5))
    iy_mid = np.argmin(np.abs(y - 0.5))

    # u(y) at x = 0.5
    u_cfd_center = u_cfd[ix_mid, :]
    u_pinn_center = U_pinn[ix_mid, :]

    # v(x) at y = 0.5
    v_cfd_center = v_cfd[:, iy_mid]
    v_pinn_center = V_pinn[:, iy_mid]

    plt.figure(figsize=(10, 4))

    # Left: u(y) at x=0.5
    plt.subplot(1, 2, 1)
    plt.plot(u_cfd_center, y, "o-", label="CFD")
    plt.plot(u_pinn_center, y, "s--", label="PINN")
    plt.gca().invert_yaxis()
    plt.xlabel("u(x=0.5, y)")
    plt.ylabel("y")
    plt.title("Vertical centerline: u(y) at x=0.5")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # Right: v(x) at y=0.5
    plt.subplot(1, 2, 2)
    plt.plot(x, v_cfd_center, "o-", label="CFD")
    plt.plot(x, v_pinn_center, "s--", label="PINN")
    plt.xlabel("x")
    plt.ylabel("v(x, y=0.5)")
    plt.title("Horizontal centerline: v(x) at y=0.5")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()


# -------------------------
#  Velocity magnitude comparison
# -------------------------

def plot_velocity_magnitude_comparison(x, y, u_cfd, v_cfd, U_pinn, V_pinn):
    """Compare |u| contours: CFD vs PINN."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    vel_cfd = np.sqrt(u_cfd**2 + v_cfd**2)
    vel_pinn = np.sqrt(U_pinn**2 + V_pinn**2)

    vmin = min(vel_cfd.min(), vel_pinn.min())
    vmax = max(vel_cfd.max(), vel_pinn.max())

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    cf1 = plt.contourf(X, Y, vel_cfd, levels=51, vmin=vmin, vmax=vmax)
    plt.colorbar(cf1, label="|u|")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("CFD: velocity magnitude")
    plt.gca().set_aspect("equal")

    plt.subplot(1, 2, 2)
    cf2 = plt.contourf(X, Y, vel_pinn, levels=51, vmin=vmin, vmax=vmax)
    plt.colorbar(cf2, label="|u|")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PINN: velocity magnitude")
    plt.gca().set_aspect("equal")

    plt.tight_layout()


# -------------------------
#  Vorticity comparison
# -------------------------

def plot_vorticity_comparison(x, y, u_cfd, v_cfd, U_pinn, V_pinn):
    """Compare vorticity contours: CFD vs PINN."""
    X, Y = np.meshgrid(x, y, indexing="ij")

    omega_cfd = compute_vorticity(x, y, u_cfd, v_cfd)
    omega_pinn = compute_vorticity(x, y, U_pinn, V_pinn)

    om_min = min(omega_cfd.min(), omega_pinn.min())
    om_max = max(omega_cfd.max(), omega_pinn.max())

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    cf1 = plt.contourf(X, Y, omega_cfd, levels=51, vmin=om_min, vmax=om_max)
    plt.colorbar(cf1, label="ω")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("CFD: vorticity")
    plt.gca().set_aspect("equal")

    plt.subplot(1, 2, 2)
    cf2 = plt.contourf(X, Y, omega_pinn, levels=51, vmin=om_min, vmax=om_max)
    plt.colorbar(cf2, label="ω")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PINN: vorticity")
    plt.gca().set_aspect("equal")

    plt.tight_layout()


# -------------------------
#  Error fields
# -------------------------

def plot_error_fields(x, y, u_cfd, v_cfd, U_pinn, V_pinn):
    """Plot |u_error| and |v_error| fields and print global L2 errors."""
    X, Y = np.meshgrid(x, y, indexing="ij")

    err_u = np.abs(u_cfd - U_pinn)
    err_v = np.abs(v_cfd - V_pinn)

    # relative L2 errors
    rel_l2_u = np.linalg.norm(u_cfd - U_pinn) / np.linalg.norm(u_cfd)
    rel_l2_v = np.linalg.norm(v_cfd - V_pinn) / np.linalg.norm(v_cfd)

    print(f"Relative L2 error (u): {rel_l2_u:.3e}")
    print(f"Relative L2 error (v): {rel_l2_v:.3e}")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    cf1 = plt.contourf(X, Y, err_u, levels=51)
    plt.colorbar(cf1, label="|u_CFD - u_PINN|")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Error in u")
    plt.gca().set_aspect("equal")

    plt.subplot(1, 2, 2)
    cf2 = plt.contourf(X, Y, err_v, levels=51)
    plt.colorbar(cf2, label="|v_CFD - v_PINN|")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Error in v")
    plt.gca().set_aspect("equal")

    plt.tight_layout()


# -------------------------
#  Main
# -------------------------

def main():
    # 1) Load PINN and CFD
    model = load_trained_model()
    x, y, u_cfd, v_cfd, p_cfd = load_cfd_results()

    # 2) Evaluate PINN on the same grid as CFD
    U_pinn, V_pinn, P_pinn = evaluate_pinn_on_grid(model, x, y)

    # 3) Training history
    plot_training_history()

    # 4) Quick PINN-only velocity field (optional)
    plot_velocity_field(model)

    # 5) Centerline comparison (u & v)
    plot_centerline_comparison(x, y, u_cfd, v_cfd, U_pinn, V_pinn)

    # 6) Velocity magnitude contours (CFD vs PINN)
    plot_velocity_magnitude_comparison(x, y, u_cfd, v_cfd, U_pinn, V_pinn)

    # 7) Vorticity comparison (CFD vs PINN)
    plot_vorticity_comparison(x, y, u_cfd, v_cfd, U_pinn, V_pinn)

    # 8) Error fields and global L2 error
    plot_error_fields(x, y, u_cfd, v_cfd, U_pinn, V_pinn)

    plt.show()


if __name__ == "__main__":
    main()
