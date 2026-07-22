import numpy as np
import warnings

warnings.simplefilter("always", Warning)

def relative_error_stats(validation, prediction, dx, dy, print_error=True, return_error=False):
    """
    Computes and prints statistical metrics for the relative error between 
    the test data (`validation`) and the predicted values (`prediction`).

    Parameters:
    - validation: Tensor or array containing reference values (ground truth).
    - prediction: Tensor or array containing predicted values.
    - dx: Step size in the x-direction for numerical integration.
    - dy: Step size in the y-direction for numerical integration.

    The function calculates the relative error using numerical integration
    and prints key statistics such as minimum, quartiles, median, and maximum.
    """

    # Convert input tensors to NumPy arrays
    validation = validation.numpy()
    prediction = prediction.numpy()

    # Compute the squared error using numerical integration
    numerator = np.trapz(np.trapz((validation - prediction)**2, dx=dy), dx=dx)
    denominator = np.trapz(np.trapz(validation**2, dx=dy), dx=dx)

    # Check if the denominator is zero to avoid division by zero
    if np.all(denominator == 0):
        warnings.warn("Denominator = 0 --> Absolute error", Warning)
        if print_error == True:
            print("Absolute error (denominator = 0):")
        prediction_error = np.sqrt(numerator)  # Compute absolute error
    else:
        if print_error == True:
            print("Relative error:")
        prediction_error = np.sqrt(numerator / denominator)  # Compute relative error

    if print_error == True:
        # Compute descriptive statistics of the error
        min = np.min(prediction_error)
        max = np.max(prediction_error)
        Q1 = np.percentile(prediction_error, 25)
        Q2 = np.percentile(prediction_error, 50)
        Q3 = np.percentile(prediction_error, 75)

        # Print error statistics
        print(f" Minimum: {min:.2e}")
        print(f" First quartile (Q1): {Q1:.2e}")
        print(f" Median (Q2): {Q2:.2e}")
        print(f" Third quartile (Q3): {Q3:.2e}")
        print(f" Maximum: {max:.2e}")

    if return_error==True:
        return prediction_error