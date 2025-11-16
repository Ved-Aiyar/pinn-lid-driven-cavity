import os
import numpy as np
import matplotlib.pyplot as plt

# Path to results file
here = os.path.dirname(__file__)
results_path = os.path.join(here, "results", "cavity_Re100.npz")


def load_cfd_results():
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"CFD results not found at {results_path}. "
            "Run cfd_solver.py first."
        )
    data = np.load(results_path)
    x = data["x"]
    y = data["y"]
    u = data["u"]
    v = data["v"]
    p = data["p"]
    return x, y, u, v, p


def compute_vorticity(x, y, u, v):
    nx, ny = u.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    omega = np.zeros_like(u)

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            dv_dx = (v[i + 1, j] - v[i - 1, j]) / (2.0 * dx)
            du_dy = (u[i, j + 1] - u[i, j - 1]) / (2.0 * dy)
            omega[i, j] = dv_dx - du_dy

    omega[0, :] = omega[1, :]
    omega[-1, :] = omega[-2, :]
    omega[:, 0] = omega[:, 1]
    omega[:, -1] = omega[:, -2]

    return omega


def plot_cfd_fields():
    x, y, u, v, p = load_cfd_results()
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)

    # Vorticity
    omega = compute_vorticity(x, y, u, v)

    # 1) Velocity magnitude contour
    plt.figure()
    cf = plt.contourf(X, Y, vel_mag, levels=51)
    plt.colorbar(cf, label="|u|")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("CFD: Velocity magnitude")
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    # 2) Pressure contour
    plt.figure()
    cf = plt.contourf(X, Y, p, levels=51)
    plt.colorbar(cf, label="p")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("CFD: Pressure")
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    # 3) Vorticity contour
    plt.figure()
    cf = plt.contourf(X, Y, omega, levels=51)
    plt.colorbar(cf, label="ω")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("CFD: Vorticity")
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    # 4) Velocity quiver (coarsened grid to avoid clutter)
    skip = max(1, len(x) // 20)
    plt.figure()
    plt.quiver(X[::skip, ::skip], Y[::skip, ::skip],
               u[::skip, ::skip], v[::skip, ::skip])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("CFD: Velocity field (quiver)")
    plt.gca().set_aspect("equal")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_cfd_fields()
