import torch
import matplotlib.pyplot as plt

def plot_system_grid(z_target: torch.Tensor, z_predicted: torch.Tensor):
    """Plots system states in a (3, 2) grid for compact comparison.

    Args:
        z_target: Ground truth tensor.
        z_predicted: Predicted TensOps object with.
    """
    real = z_target.detach().cpu().numpy()
    pred = z_predicted.values.squeeze().detach().cpu().numpy()

    # (idx_1, idx_2, Label_1, Label_2, Title)
    plot_configs = [
        (0, 1, "$q_1(x)$", "$q_1(y)$", "Coordinates $q_1$"),
        (2, 3, "$q_2(x)$", "$q_2(y)$", "Coordinates $q_2$"),
        (4, 5, "$p_1(x)$", "$p_1(y)$", "Momenta $p_1$"),
        (6, 7, "$p_2(x)$", "$p_2(y)$", "Momenta $p_2$"),
        (8, 9, "$e_1$", "$e_2$", "$e_i$")
    ]

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, (idx1, idx2, lbl1, lbl2, title) in enumerate(plot_configs):
        ax = axes[i]
        
        # Plot first component
        ax.plot(real[:, idx1], color="black", linestyle="--", alpha=1)
        ax.plot(pred[:, idx1], label=f"SPNN {lbl1}", color="#1f77b4", linewidth=1)
        
        # Plot second component
        ax.plot(real[:, idx2], label=f"GT", color="black", linestyle="--", alpha=1)
        ax.plot(pred[:, idx2], label=f"SPNN {lbl2}", color="#ff7f0e", linewidth=1)

        ax.set_xlabel("Time Steps")
        ax.set_ylabel(title)
        ax.legend(loc="best", fontsize="small", ncol=2)
        ax.grid(True, linestyle=":", alpha=0.7)

    axes[5].axis("off") # Clean up the 6th subplot

    plt.tight_layout()
    plt.show()