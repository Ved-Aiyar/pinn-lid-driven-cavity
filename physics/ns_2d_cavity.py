import torch


def compute_ns_residuals(model, x, y, nu):
    """Compute PDE residuals for 2D steady incompressible Navier-Stokes.

    model: PINN_NS2D, outputs (u, v, p)
    x, y: tensors of shape (N, 1) with requires_grad=True
    nu: kinematic viscosity (1/Re)

    Returns:
        r_cont: continuity residual (N, 1)
        r_mom_x: x-momentum residual (N, 1)
        r_mom_y: y-momentum residual (N, 1)
    """

    # Concatenate inputs
    xy = torch.cat([x, y], dim=1)  # (N, 2)

    # Forward pass
    out = model(xy)
    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    # First-order derivatives
    grads_u = torch.autograd.grad(
        u, [x, y],
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )
    u_x, u_y = grads_u[0], grads_u[1]

    grads_v = torch.autograd.grad(
        v, [x, y],
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True,
    )
    v_x, v_y = grads_v[0], grads_v[1]

    grads_p = torch.autograd.grad(
        p, [x, y],
        grad_outputs=torch.ones_like(p),
        create_graph=True,
       retain_graph=True,
    )
    p_x, p_y = grads_p[0], grads_p[1]

    # Second-order derivatives
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_yy = torch.autograd.grad(
        u_y, y,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True,
        retain_graph=True,
    )[0]

    v_xx = torch.autograd.grad(
        v_x, x,
        grad_outputs=torch.ones_like(v_x),
        create_graph=True,
        retain_graph=True,
    )[0]
    v_yy = torch.autograd.grad(
        v_y, y,
        grad_outputs=torch.ones_like(v_y),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Continuity residual
    r_cont = u_x + v_y

    # Momentum residuals
    conv_x = u * u_x + v * u_y
    conv_y = u * v_x + v * v_y

    diff_u = nu * (u_xx + u_yy)
    diff_v = nu * (v_xx + v_yy)

    r_mom_x = conv_x + p_x - diff_u
    r_mom_y = conv_y + p_y - diff_v

    return r_cont, r_mom_x, r_mom_y
