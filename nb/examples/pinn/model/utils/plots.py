import numpy as np
import matplotlib.pyplot as plt

def plot_model_comparison(t_analytic, x_analytic, model_ag, model_fd, t_data, x_data, collocation_points):
    """
    Visualizes the performance of Autograd vs Finite-Difference models.
    
    Args:
        t_analytic: TensOps containing time values for evaluation.
        x_analytic: TensOps containing the ground truth solution.
        model_ag: The model using Autograd.
        model_fd: The model using Finite-Differences.
        t_data: Observed training time points.
        x_data: Observed training state values.
        collocation_points: Points where the physics residuals are evaluated.
    """
    
    # Prediction and data extraction
    # We perform all tensor/dataframe operations here to keep the plotting logic clean
    t_plot = t_analytic.values.squeeze()
    x_true = x_analytic.values.squeeze()
    
    x_ag = model_ag(t_analytic.values).detach().squeeze()
    x_fd = model_fd(t_analytic.values).detach().squeeze()

    # Setup figure
    plt.figure(figsize=(7, 5))

    # Plot ground truth and predictions
    plt.plot(t_plot, x_true, color='steelblue', lw=2, 
             label="Analytic Solution", zorder=1)
    
    plt.plot(t_plot, x_ag, color='red', linestyle='--', lw=2, 
             label="Pred: Autograd", zorder=2)
    
    plt.plot(t_plot, x_fd, color='tab:green', linestyle=':', lw=2.5, 
             label="Pred: Finite-Differences", zorder=3)

    # Plot collocation points and data
    y_range_offset = (x_true.max() - x_true.min()) * 0.1
    y_marker_pos = x_true.min() - y_range_offset

    plt.plot(collocation_points, np.full_like(collocation_points, y_marker_pos), 
             '|', color="black", markersize=12, alpha=0.6, 
             label='Collocation Points', zorder=4)

    plt.scatter(t_data, x_data, color="tab:orange", s=40,
                label=f"Training Data ({len(t_data)} pts)", zorder=5)

    # Axis styling
    plt.xlabel("$t$", fontsize=12)
    plt.ylabel("$x(t)$", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best', frameon=True, shadow=False, fontsize=10)
    
    plt.tight_layout()
    plt.show()