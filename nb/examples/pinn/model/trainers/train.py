import torch
import torch.nn as nn
import tensopsciml as to

from tensopsciml.tensors import TensOps, MetrOps

def train_pinn_finite_differences(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    t_data: torch.Tensor,
    x_data: torch.Tensor,
    t_physics_: torch.Tensor,
    mu: float,
    k: float,
    D,
    inv_metric: MetrOps,
    e_t: TensOps,
    iterations: int = 50000,
    data_weight: float = 1e4
) -> None:
    """Trains a Physics-Informed Neural Network for a damped oscillator.

    Args:
        model: The neural network (MLP) approximating the solution x(t).
        t_data: Time coordinates for the labeled experimental data.
        x_data: Observed values at t_data.
        t_physics_: Collocation points for evaluating the physics residual.
        mu: Damping coefficient.
        k: Spring constant.
        D: Stack of first-derivative kernels on the time axis.
        inv_metric: Inverse metric g^{ab} of the time domain (order [2, 0]),
            on the grid of the first partial derivative (one node smaller
            than the collocation grid).
        e_t: Coordinate basis vector of the time axis (order [1, 0]), on the
            same grid as inv_metric.
        iterations: Number of optimization steps.
        data_weight: Scalar multiplier to balance data vs. physics loss.
    """

    for i in range(iterations):
        optimizer.zero_grad()

        # Data Loss
        x_pred_data = model(t_data)
        loss_data = torch.mean((x_pred_data - x_data)**2)

        # Compute ODE (x'' + mu*x' + k*x = 0)
        x_pred_phys_ = model(t_physics_).squeeze().unsqueeze(0).unsqueeze(0)
        x_pred_phys = TensOps(
            x_pred_phys_, 
            domain_dim=1, 
            codomain_dim=[1], 
            order=[0, 0], 
            contravariant_indices=None, 
            covariant_indices=None
        )
        # d_a x is COVARIANT (the differential of a scalar).
        dx = to.partial_derivative(x_pred_phys, D)  # d_a x [0, 1]

        # First-order term: the directional derivative along the time
        # coordinate basis vector, e^a d_a x.
        dx_dt = to.contraction(e_t, dx, [[0], [None]], [[None], [0]])  # [0, 0]

        # Second-order term: the same directional derivative applied twice,
        # e^a d_a(e^b d_b x). 
        d_dx_dt = to.partial_derivative(dx_dt, D)  # d_c (dx/dt) [0, 1]
        d2x_dt2 = to.contraction(
            to.m(e_t), d_dx_dt, [[0], [None]], [[None], [0]]
        )  # [0, 0]

        # Physics loss. Every term is now a scalar; m() only realigns the grids
        # that each finite-difference stencil shrinks by one node.
        physics_res = d2x_dt2 + mu * to.m(dx_dt) + k * to.m(to.m(x_pred_phys))
        loss_physics = torch.mean(physics_res.values**2)

        # Total Loss and optimization
        total_loss = data_weight * loss_data + loss_physics
        total_loss.backward()
        optimizer.step()

        # Logs
        if (i + 1) % 10000 == 0:
            print(
                f"Iter {i+1:6d} | "
                f"Data: {loss_data.item():.3e} | "
                f"Phys: {loss_physics.item():.3e} | "
                f"Total: {total_loss.item():.3e}"
            )


def train_pinn_finite_autograd(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    t_data: torch.Tensor,
    x_data: torch.Tensor,
    t_physics: TensOps,
    mu: float,
    k: float,
    inv_metric: MetrOps,
    e_t: TensOps,
    iterations: int = 50000,
    data_weight: float = 1e4
) -> None:
    """Trains a Physics-Informed Neural Network for a damped oscillator.

    Args:
        model: The neural network (MLP) approximating the solution x(t).
        t_data: Time coordinates for the labeled experimental data.
        x_data: Observed values at t_data.
        t_physics: Collocation points as the contravariant time coordinate
            t^a (order [1, 0]); differentiating w.r.t. it lowers indices.
        mu: Damping coefficient.
        k: Spring constant.
        inv_metric: Inverse metric g^{ab} of the time domain (order [2, 0]),
            on the collocation grid (autograd does not shrink it).
        e_t: Coordinate basis vector of the time axis (order [1, 0]), on the
            same grid as inv_metric.
        iterations: Number of optimization steps.
        data_weight: Scalar multiplier to balance data vs. physics loss.
    """

    for i in range(iterations):
        optimizer.zero_grad()

        # Data Loss
        x_pred_data = model(t_data)
        loss_data = torch.mean((x_pred_data - x_data)**2)

        # Compute ODE (x'' + mu*x' + k*x = 0)
        x_pred_phys_ = model(t_physics.values)
        x_pred_phys = TensOps(
            x_pred_phys_,
            domain_dim=1,
            codomain_dim=[1],
            order=[0, 0],
            contravariant_indices=None,
            covariant_indices=None
        )

        # d_a x is COVARIANT (the differential of a scalar).
        dx = to.autograd(x_pred_phys, t_physics, create_graph=True)  # d_a x [0, 1]

        # First-order term: the directional derivative along the time
        # coordinate basis vector, e^a d_a x.
        dx_dt = to.contraction(e_t, dx, [[0], [None]], [[None], [0]])  # [0, 0]

        # Second-order term: the same directional derivative applied twice,
        # e^a d_a(e^b d_b x). 
        d_dx_dt = to.autograd(dx_dt, t_physics, create_graph=True)  # d_c (dx/dt) [0, 1]
        d2x_dt2 = to.contraction(e_t, d_dx_dt, [[0], [None]], [[None], [0]])  # [0, 0]

        # Physics loss. Every term is now a scalar on the same grid.
        physics_res = d2x_dt2 + mu * dx_dt + k * x_pred_phys
        loss_physics = torch.mean(physics_res.values**2)

        # Total Loss and optimization
        total_loss = data_weight * loss_data + loss_physics
        total_loss.backward()
        optimizer.step()

        # Logs
        if (i + 1) % 10000 == 0:
            print(
                f"Iter {i+1:6d} | "
                f"Data: {loss_data.item():.3e} | "
                f"Phys: {loss_physics.item():.3e} | "
                f"Total: {total_loss.item():.3e}"
            )