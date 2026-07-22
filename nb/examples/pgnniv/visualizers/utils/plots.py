import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_tensors(list_of_tensors, titles, height=3, tick_fontsize=12):  
    """
    Plots a list of tensors using matplotlib.

    Parameters:
    - list_of_tensors (list of 2D arrays/tensors): List of tensors or arrays to be visualized.
    - titles (list of str): Titles for each subplot.
    - height (int, optional): Height of each row in inches. Default is 3.
    - tick_fontsize (int, optional): Font size for axis ticks. Default is 12.

    The function dynamically arranges subplots in a grid, adds color bars, and formats tick labels.
    """

    if len(list_of_tensors) != len(titles):
        raise ValueError("Number of titles and tensors don't match.")
    
    num_plots = len(list_of_tensors)
    
    # Handle case when there is only one tensor
    if num_plots == 1:
        cols = 1
        rows = 1
    else:
        cols = min(num_plots, 2)
        rows = (num_plots + 1) // 2

    # Create subplots with dynamic figure size
    fig, axes = plt.subplots(rows, cols, figsize=(height * 2.4, rows * height))  
    if num_plots == 1:
        axes = np.array([[axes]])
    else:
        axes = np.array(axes).reshape(rows, cols)  

    # Iterate through tensors and plot each one
    for i, diff in enumerate(list_of_tensors):  
        row, col = divmod(i, cols)  
        im = axes[row, col].imshow(diff.squeeze(), interpolation='bicubic', extent=[0, 1, 0, 1]) 

        cbar = plt.colorbar(im, ax=axes[row, col], fraction=0.04, pad=0.02)  
        cbar.formatter = ScalarFormatter(useMathText=True)  
        cbar.formatter.set_powerlimits((0, 0))  
        cbar.update_ticks()  

        ticks = cbar.get_ticks()  
        cbar.set_ticks(ticks[1:-1] + 0.00) # To adjust notation in the colorbar ticks
        axes[row, col].tick_params(axis='both', which='major', labelsize=tick_fontsize)  
        axes[row, col].set_title(titles[i], fontsize=15)  

    if num_plots % 2 != 0 and cols == 2:  
        axes[-1, -1].axis("off")  

    plt.tight_layout()
    plt.show()