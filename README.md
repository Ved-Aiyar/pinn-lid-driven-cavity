# Physics-Informed Neural Network (PINN) Solution of the 2D Lid-Driven Cavity Flow

This project implements a Physics-Informed Neural Network (PINN) to solve the 2D incompressible Navier–Stokes equations for the lid-driven cavity flow benchmark problem.
To validate the PINN’s accuracy, results are compared with a Finite Difference Method (FDM) CFD solver implemented separately.

Key Features:
-   PINN architecture for solving Navier–Stokes equations
-   Uses automatic differentiation to enforce PDE residuals
-   Boundary condition enforcement for lid-driven cavity
-   Classical CFD solver (FDM) for reference
-   Post-processing tools for side-by-side comparison
-   Error metrics for measuring PINN accuracy

Project Structure:

pinn-lid-driven-cavity/
│
├── main_train.py          # Training loop + checkpointing
├── postprocess.py         # PINN vs CFD comparison + plots
│
├── models/
│   └── pinn_ns.py         # Neural network (PINN) architecture
│
├── physics/
│   └── ns_2d_cavity.py    # Navier-Stokes PDE residual computation
│
├── data/
│   └── sampling.py        # Collocation + boundary point sampling
│
├── utils/
│   ├── plotting.py        # Visualization utilities
│   └── error_metrics.py   # Error computation (L2 norm)
│
├── cfd/
│   ├── cfd_solver.py      # Finite difference CFD solver
│   └── results/           # CFD results (.npz)
│
├── checkpoints/           # Saved PINN model weights
└── README.md              # Project documentation


Mathematical Formulation:
The PINN solves the steady 2D incompressible Navier–Stokes equations.
1. ∇⋅u=0
2. (u⋅∇)u=−∇p+ν∇2u
Boundary conditions for the unit cavity:
1. Lid (top wall): u=1,v=0
2. Other walls: u=0,v=0


How to Run:
1.  Create virtual environment
2.  Install dependencies
3.  Train model
4.  Run post-process

CFD Solver:
A simple finite difference method is implemented to solve the same cavity flow problem at Re = 100.
The solver computes:
1.  u(x,y),v(x,y)
2.  p(x,y)
3.  Grid values
4.  Stored in .npz format under cfd/results/  

Importance:
Demonstrates ML + CFD integration, PINN accuracy, and numerical methods.
