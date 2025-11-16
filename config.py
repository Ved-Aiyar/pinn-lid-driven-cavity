import torch

# Domain settings
DOMAIN_X_MIN = 0.0
DOMAIN_X_MAX = 1.0
DOMAIN_Y_MIN = 0.0
DOMAIN_Y_MAX = 1.0

# Physics
REYNOLDS_NUMBER = 100.0  # adjust as needed
NU = 1.0 / REYNOLDS_NUMBER  # kinematic viscosity (non-dimensional)

# Lid velocity (top wall)
U_LID = 1.0

# Training settings
N_COLLOCATION = 5000   # interior points
N_BOUNDARY = 2000      # boundary points total

N_EPOCHS = 5000
LEARNING_RATE = 1e-3

# Network
INPUT_DIM = 2    # (x, y)
OUTPUT_DIM = 3   # (u, v, p)
HIDDEN_DIM = 64
N_HIDDEN_LAYERS = 8

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"