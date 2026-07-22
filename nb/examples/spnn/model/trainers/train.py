import torch
import torch.nn as nn

from tensopsciml.tensors import TensOps
from typing import List, Dict, Tuple

def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_indices: List[int],
    z_dataset: torch.Tensor,
    l_matrix: TensOps,
    m_matrix: TensOps,
    dt: float,
    max_epochs: int = 5000,
    data_weight: float = 1e4,
    log_interval: int = 1000
) -> Dict[str, List[float]]:
    """Trains the SPNN model using gradient accumulation over trajectories.

    Args:
        model: The SPNN neural network instance.
        optimizer: PyTorch optimizer.
        train_indices: List of indices for the training trajectories.
        z_dataset: The full state dataset [N_traj, Dim, Time].
        l_matrix: Poisson tensor L^{ij} (order [2, 0]).
        m_matrix: Friction tensor M^{ij} (order [2, 0]).
        dt: Time step size.
        max_epochs: Total number of training iterations.
        data_weight: Scalar to balance data loss vs degeneracy loss.
        log_interval: How often to print progress to the console.

    Returns:
        A dictionary containing historical loss values for analysis.
    """
    history = {
        "epoch": [],
        "loss_data": [],
        "loss_degeneracy": []
    }

    model.train()
    print(f"-- Starting training for {max_epochs} epochs -- ")

    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        
        epoch_data_loss = 0.0
        epoch_deg_loss = 0.0
        num_trajectories = len(train_indices)

        for traj_idx in train_indices:

            # Extract ground truth: Slice time to get (t) and (t+1)
            z_gt = z_dataset[traj_idx][:, :-1].T  # Input states
            z1_gt = z_dataset[traj_idx][:, 1:].T  # Target next states

            # Forward pass
            z1_out, deg_e, deg_s = model(z_gt, l_matrix, m_matrix, dt)

            # Loss calculation (using mean for stability)
            loss_data = torch.mean((z1_out.values.squeeze() - z1_gt)**2)
            loss_deg = torch.mean(deg_e.values**2) + torch.mean(deg_s.values**2)

            total_loss = (data_weight * loss_data) + loss_deg
            total_loss.backward()

            epoch_data_loss += loss_data.item()
            epoch_deg_loss += loss_deg.item()

        # Update weights
        optimizer.step()

        # Averaging losses and logging
        mean_data = epoch_data_loss / num_trajectories
        mean_deg = epoch_deg_loss / num_trajectories

        history["epoch"].append(epoch)
        history["loss_data"].append(mean_data)
        history["loss_degeneracy"].append(mean_deg)

        if epoch == 1 or epoch % log_interval == 0:
            print(f"Epoch {epoch:4d}/{max_epochs} | "
                  f"Data Loss: {mean_data:.2e} | "
                  f"Deg. Loss: {mean_deg:.2e}")

    return history