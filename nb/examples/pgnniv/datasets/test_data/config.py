import os

BASE_DIR = os.path.abspath(__file__)

# Configure data name and the absolute path where they will be saved
N_DATA = 100
NOISE = 0
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', 'datasets/test_data/data'))

# Each test set sweeps lambda1 over [1.0, 1.1] at a FIXED transverse stretch
# lambda2, giving one stress-strain curve per value. The validation notebook
# loads all three by name, so generating fewer leaves it unable to run.
LAMBDA2_VALUES = [1.02, 1.05, 1.08]


def dataset_name(lambda2):
    """Dataset file name for a given fixed lambda2 (1.05 -> ..._test_lambda105)."""
    return f'Ogden_N{N_DATA}_S{NOISE}_test_lambda{int(round(lambda2 * 100))}'


def dataset_config(lambda2):
    """Build the generator config for one fixed transverse stretch lambda2."""
    return {
        "PATH": DATA_PATH,                     # Directory path to save generated data
        "DATASET_NAME": dataset_name(lambda2), # Name of the dataset
        "N_DATA": N_DATA,                      # Number of data points to generate
        "N_DISCRETIZATION_x": 11,              # Number of discretization points along the x-axis
        "N_DISCRETIZATION_y": 9,               # Number of discretization points along the y-axis
        "x0": 0,                               # Start of range for x-axis
        "xN": 1,                               # End of range for x-axis
        "y0": 0,                               # Start of range for y-axis
        "yN": 1,                               # End of range for y-axis
        "noise_sigma": NOISE,                  # Percentage of noise added
        "lambda2": lambda2,                    # Fixed transverse stretch for this test set
        "custom_seed": 42                      # Custom seed for reproducibility
    }
