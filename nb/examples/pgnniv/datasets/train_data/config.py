import os

BASE_DIR = os.path.abspath(__file__)

# Configure data name and the absolute path where they will be saved
N_DATA = 1000
NOISE = 0
DATASET_NAME = f'Ogden_N{N_DATA}_S{NOISE}_train'
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', 'datasets/train_data/data'))

# Save configuration in a dictionary
DATASET_CONFIG = {
    "PATH": DATA_PATH,             # Directory path to save generated data
    "DATASET_NAME": DATASET_NAME,  # Name of the dataset
    "N_DATA": N_DATA,              # Number of data points to generate
    "N_DISCRETIZATION_x": 11,      # Number of discretization points along the x-axis
    "N_DISCRETIZATION_y": 9,       # Number of discretization points along the y-axis
    "x0": 0,                       # Start of range for x-axis
    "xN": 1,                       # End of range for x-axis
    "y0": 0,                       # Start of range for y-axis
    "yN": 1,                       # End of range for y-axis
    "noise_sigma": NOISE,          # Percentage of noise added
    "custom_seed": 42              # Custom seed for reproducibility     
}