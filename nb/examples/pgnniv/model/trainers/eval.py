import torch

import tensopsciml as to
from tensopsciml.tensors import TensOps
from model.utils.tractions import compute_boundary_tractions

# Loss weights. Every residual is non-dimensionalized by its own characteristic
# scale inside `loss_function`, so all four MSEs are RELATIVE errors of order 1
# and these weights are directly comparable.
W_R1 = 10.0
W_R2 = 0.01
W_R3 = 1.00
W_R4 = 10.0

# Characteristic length of the reference domain.
L_DOMAIN = 1.0

def restriction_R1(U_true, U_pred):
    """
    Compute the residual between `U_true` and `U_pred`.

    Parameters:
        U_true (ndarray or tensor): The ground truth values for U.
        U_pred (ndarray or tensor): The predicted values for U.

    Returns:
        ndarray or tensor: The residual computed as `U_true - U_pred`.
    """
    R1 = (U_true - U_pred)
    return R1

def restriction_R2(P_pred, D):
    """
    Reference-configuration equilibrium residual (Div P)^i = d_j P^{ij}.

    P is the first Piola-Kirchhoff stress P^{ij} [2, 0] (twice contravariant),
    so its reference divergence needs neither a metric nor index raising:
    differentiate to get d_k P^{ij}, then contract the covariant derivative index
    k with P's reference contravariant index j.

    Parameters:
        P_pred (TensOps): Predicted first Piola-Kirchhoff stress P^{ij} [2, 0].
        D (torch.Tensor): Derivative filters.

    Returns:
        tuple: The two components (R2x, R2y) of Div P (weighted to zero below).
    """
    dP = to.partial_derivative(P_pred, D.to(P_pred.values.device))
    div_P = to.self_contraction(dP, [1], [0]) 
    R2x = div_P.values[:, 0]
    R2y = div_P.values[:, 1]
    return R2x, R2y

def restriction_R3(P_pred, Tx_true, Ty_true):
    """
    Compute the traction boundary-condition residuals via T^i = P^{ij} N_j.

    The predicted tractions are the full first Piola-Kirchhoff traction VECTORS
    on the positive x and y faces (from ``compute_boundary_tractions``). The
    ground-truth Tx / Ty are the corresponding full traction vectors measured on
    those same faces, so the residual penalizes both the normal and the
    tangential (shear) components directly.

    Parameters:
        P_pred (TensOps): Predicted first Piola-Kirchhoff stress P^{ij} [2, 0].
        Tx_true (torch.Tensor): Ground-truth traction vector on the x-face,
            shape (N, 2, Ny, 1).
        Ty_true (torch.Tensor): Ground-truth traction vector on the y-face,
            shape (N, 2, 1, Nx).

    Returns:
        tuple: Traction residual tensors ``(R3x, R3y)`` for the x- and y-faces.
    """
    tx_pred, ty_pred = compute_boundary_tractions(P_pred)

    R3x = tx_pred.values - Tx_true
    R3y = ty_pred.values - Ty_true
    return R3x, R3y

def restriction_R4(U_pred):
    """
    Extract specific slices or components from `U_pred`.

    Parameters:
        U_pred (object): A multidimensional array or tensor containing predicted U values.

    Returns:
        tuple: Two components `(R4x, R4y)` where:
            - `R4x`: Values from the first dimension at specific indices.
            - `R4y`: Values from the second dimension at specific indices.
    """
    R4x = U_pred.values[:, 0, :, 0] 
    R4y = U_pred.values[:, 1, 0, :]
    return R4x, R4y
    
def MSE(diff_tensor):
    """
    Computes the sum of the squares of all components in the given tensor.

    Parameters:
        diff_tensor (torch.Tensor or TensOps): Object containing the difference values.

    Returns:
        (torch.Tensor): The MSE value, defined as the sum of squared components in the tensor.

    Raises:
        ValueError: If `diff_tensor` is neither a torch.Tensor nor a TensOps object.
    """

     # For regular PyTorch tensors, directly compute the mean of squares.
    if isinstance(diff_tensor, torch.Tensor):
        return torch.mean(torch.square(diff_tensor))

    # For TensOps objects, access the '.values' and compute the mean of squares.
    elif isinstance(diff_tensor, TensOps):
        return torch.mean(torch.square(diff_tensor.values))

    # In other case, raise a ValueError
    else:
        raise ValueError("Not compatible data type. Need torch.Tensor or TensOps object.")
    
def loss_function(U_train, U_pred, P_pred, Tx_train, Ty_train, D):
    """
    Computes the total loss as a weighted sum of constraints (e, pi1, pi2, pi3).

    Parameters:
        U_train (TensOps): Ground-truth displacement field.
        U_pred (TensOps): Predicted displacement field.
        P_pred (TensOps): Predicted first Piola-Kirchhoff stress P^{ij} [2, 0].
        Tx_train (torch.Tensor): Ground-truth traction vector on the x-face.
        Ty_train (torch.Tensor): Ground-truth traction vector on the y-face.
        D (torch.Tensor): Derivative filters.

    Returns:
        total_loss (float): The total loss, computed as a weighted sum of individual constraint losses.
        e (float): Loss contribution from the `e_constraint`.
        pi1 (float): Loss contribution from the `pi1_constraint`.
        pi2 (float): Loss contribution from the `pi2_constraint`.
        pi3 (float): Loss contribution from the `pi3_constraint`.
    """
    
    # Compute the MSE for each individual constraint
    R1 = restriction_R1(U_train, U_pred)
    R2x, R2y = restriction_R2(P_pred, D)
    R3x, R3y = restriction_R3(P_pred, Tx_train, Ty_train)
    R4x, R4y = restriction_R4(U_pred)

    # Characteristic scales of the two physical quantities involved, taken from
    # the ground truth in this batch.
    u_scale = U_train.values.abs().mean().clamp_min(1e-12)
    t_scale = torch.cat([Tx_train.norm(dim=1).flatten(),
                         Ty_train.norm(dim=1).flatten()]).mean().clamp_min(1e-12)

    # Equilibrium is the odd one out: Div P is a stress GRADIENT, so it scales as
    # t_scale / L_DOMAIN.
    div_scale = (t_scale / L_DOMAIN).clamp_min(1e-12)

    MSE_R1 = MSE(R1 / u_scale)
    MSE_R2x, MSE_R2y = MSE(R2x / div_scale), MSE(R2y / div_scale)
    MSE_R3 = MSE(R3x / t_scale) + MSE(R3y / t_scale)
    MSE_R4x, MSE_R4y = MSE(R4x / u_scale), MSE(R4y / u_scale)

    # Combine individual losses into a total loss with specific weighting factors.
    total_loss = (W_R1 * MSE_R1
                  + W_R2 * (MSE_R2x + MSE_R2y)
                  + W_R3 * MSE_R3
                  + W_R4 * (MSE_R4x + MSE_R4y))

    # Return the total loss and individual components.
    return total_loss, MSE_R1, MSE_R2x, MSE_R2y, MSE_R3, MSE_R4x, MSE_R4y