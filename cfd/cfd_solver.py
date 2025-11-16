import numpy as np
import os

# -----------------------
# Configuration
# -----------------------

Re = 100.0          # Reynolds number (must match your PINN)
nu = 1.0 / Re       # kinematic viscosity
rho = 1.0           # density
U_lid = 1.0         # lid velocity

Lx = 1.0
Ly = 1.0

nx = 41             # grid points in x
ny = 41             # grid points in y
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

dt = 0.001          # time step
nt = 5000           # number of time steps
n_p_iter = 50       # pressure Poisson iterations per time step

output_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "cavity_Re100.npz")


# -----------------------
# Helper functions
# -----------------------

def build_rhs(u, v, rho, dt, dx, dy):
    """
    Build RHS of pressure Poisson equation for incompressible flow.
    """
    b = np.zeros_like(u)

    du_dx = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
    dv_dy = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)

    du_dy = (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dy)
    dv_dx = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dx)

    b[1:-1, 1:-1] = rho * (
        (du_dx + dv_dy) / dt
        - du_dx**2
        - 2.0 * du_dy * dv_dx
        - dv_dy**2
    )

    return b


def pressure_poisson(p, b, dx, dy, n_iter):
    """
    Jacobi-type iteration to solve pressure Poisson equation:
    ∇²p = b
    with Neumann/Dirichlet boundary conditions typical of cavity flow.
    """
    pn = np.empty_like(p)

    coeff_x = dy**2
    coeff_y = dx**2
    denom = 2.0 * (dx**2 + dy**2)

    for _ in range(n_iter):
        pn[:, :] = p[:, :]

        p[1:-1, 1:-1] = (
            coeff_x * (pn[2:, 1:-1] + pn[:-2, 1:-1])
            + coeff_y * (pn[1:-1, 2:] + pn[1:-1, :-2])
            - b[1:-1, 1:-1] * dx**2 * dy**2
        ) / denom

        # Pressure boundary conditions:
        # dp/dn = 0 on all walls (Neumann)
        p[:, 0] = p[:, 1]       # bottom
        p[:, -1] = p[:, -2]     # top
        p[0, :] = p[1, :]       # left
        p[-1, :] = p[-2, :]     # right

    return p


def apply_velocity_bc(u, v, U_lid):
    """
    Apply lid-driven cavity BCs:
      - Top wall (y=1): u = U_lid, v = 0
      - Other walls: u = 0, v = 0
    """
    # Bottom wall
    u[:, 0] = 0.0
    v[:, 0] = 0.0

    # Top wall (lid)
    u[:, -1] = U_lid
    v[:, -1] = 0.0

    # Left wall
    u[0, :] = 0.0
    v[0, :] = 0.0

    # Right wall
    u[-1, :] = 0.0
    v[-1, :] = 0.0

    return u, v


# -----------------------
# Main solver
# -----------------------

def solve_cavity():
    # Coordinate arrays
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)

    # Field arrays
    u = np.zeros((nx, ny))
    v = np.zeros((nx, ny))
    p = np.zeros((nx, ny))

    u, v = apply_velocity_bc(u, v, U_lid)

    for n in range(1, nt + 1):
        un = u.copy()
        vn = v.copy()

        # Build RHS for Poisson
        b = build_rhs(un, vn, rho, dt, dx, dy)

        # Solve for pressure
        p = pressure_poisson(p, b, dx, dy, n_p_iter)

        # Velocity update (projection step)
        u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1])
            - dt / dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2])
            - dt / (2.0 * rho * dx) * (p[2:, 1:-1] - p[:-2, 1:-1])
            + nu * dt * (
                (un[2:, 1:-1] - 2.0 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / dx**2
                + (un[1:-1, 2:] - 2.0 * un[1:-1, 1:-1] + un[1:-1, :-2]) / dy**2
            )
        )

        v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - dt / dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1])
            - dt / dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2])
            - dt / (2.0 * rho * dy) * (p[1:-1, 2:] - p[1:-1, :-2])
            + nu * dt * (
                (vn[2:, 1:-1] - 2.0 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]) / dx**2
                + (vn[1:-1, 2:] - 2.0 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) / dy**2
            )
        )

        # Apply BCs
        u, v = apply_velocity_bc(u, v, U_lid)

        if n % 500 == 0 or n == 1:
            max_div = np.max(
                np.abs(
                    (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dx)
                    + (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dy)
                )
            )
            print(
                f"Step {n}/{nt}: max divergence ≈ {max_div:.3e}"
            )

    # Save results
    np.savez(
        output_path,
        x=x,
        y=y,
        u=u,
        v=v,
        p=p,
        Re=Re,
        nu=nu,
        U_lid=U_lid,
    )
    print(f"\nSaved CFD results to: {output_path}")


if __name__ == "__main__":
    solve_cavity()
