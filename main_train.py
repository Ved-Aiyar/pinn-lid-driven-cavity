import os
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DEVICE,
    NU,
    N_COLLOCATION,
    N_BOUNDARY,
    N_EPOCHS,
    LEARNING_RATE,
    INPUT_DIM,
    OUTPUT_DIM,
    HIDDEN_DIM,
    N_HIDDEN_LAYERS,
)
from models.pinn_ns import PINN_NS2D
from physics.ns_2d_cavity import compute_ns_residuals
from data.sampling import sample_interior, sample_boundary
from utils.plotting import plot_velocity_field

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "pinn_ns_cavity.pth")


def train_model():
    torch.manual_seed(42)

    model = PINN_NS2D(
        in_dim=INPUT_DIM,
        out_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_hidden_layers=N_HIDDEN_LAYERS,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()

        # To store loss history
    loss_history = {
        "epoch": [],
        "total": [],
        "pde": [],
        "bc": [],
    }


    # Sample training points once
    x_c, y_c = sample_interior(N_COLLOCATION)
    x_b, y_b, u_b, v_b = sample_boundary(N_BOUNDARY)

    x_c.requires_grad_(True)
    y_c.requires_grad_(True)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        # PDE residuals
        r_cont, r_mom_x, r_mom_y = compute_ns_residuals(model, x_c, y_c, NU)

        loss_pde = (
            mse_loss(r_cont, torch.zeros_like(r_cont))
            + mse_loss(r_mom_x, torch.zeros_like(r_mom_x))
            + mse_loss(r_mom_y, torch.zeros_like(r_mom_y))
        )

        # Boundary loss
        xy_b = torch.cat([x_b, y_b], dim=1)
        out_b = model(xy_b)
        u_pred_b = out_b[:, 0:1]
        v_pred_b = out_b[:, 1:2]

        loss_bc = mse_loss(u_pred_b, u_b) + mse_loss(v_pred_b, v_b)

        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()

        # Store losses
        loss_history["epoch"].append(epoch)
        loss_history["total"].append(loss.item())
        loss_history["pde"].append(loss_pde.item())
        loss_history["bc"].append(loss_bc.item())


        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{N_EPOCHS} "
                f"Total: {loss.item():.4e} "
                f"PDE: {loss_pde.item():.4e} "
                f"BC: {loss_bc.item():.4e}"
            )

    # Save trained model
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nSaved trained model to {CHECKPOINT_PATH}")

    # Save loss history
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.pth")
    torch.save(loss_history, history_path)
    print(f"Saved training history to {history_path}")

    return model


def load_model():
    """Create the model and load weights from checkpoint."""
    model = PINN_NS2D(
        in_dim=INPUT_DIM,
        out_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_hidden_layers=N_HIDDEN_LAYERS,
    ).to(DEVICE)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Loaded trained model from {CHECKPOINT_PATH}")
    return model


def main():
    if os.path.exists(CHECKPOINT_PATH):
        print("Checkpoint found. Loading model and skipping training.")
        model = load_model()
    else:
        print("No checkpoint found. Training model from scratch...")
        model = train_model()

    # Post-processing / plotting
    plot_velocity_field(model)


if __name__ == "__main__":
    main()
