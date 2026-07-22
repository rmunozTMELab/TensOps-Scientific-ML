import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import numpy as np
import random
import torch

from utils.folders import create_folder

class DataGenerator:
    """
    A class to generate synthetic datasets for machine learning models based on a Ogden material model.
    The datasets include functions and related data for training and validation.
    """
    def __init__(self, config, seed=42):
        """
        Initializes the DataGenerator instance with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing settings for data generation.
            seed (int, optional): Seed for random number generation (default is 42).
        """
        self.PATH = config['PATH']
        self.MODEL_NAME = config['DATASET_NAME']  
        self.N_DATA = config['N_DATA']
        self.N_DISCRETIZATION_x, self.N_DISCRETIZATION_y = config['N_DISCRETIZATION_x'], config['N_DISCRETIZATION_y'] 
        self.x0, self.xN = config['x0'], config['xN'] 
        self.y0, self.yN = config['y0'], config['yN']  
        self.noise_sigma = config['noise_sigma']/100 
        self.custom_seed = seed  
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.x, self.x_step_size = self._create_axis(self.x0, self.xN, self.N_DISCRETIZATION_x)
        self.y, self.y_step_size = self._create_axis(self.y0, self.yN, self.N_DISCRETIZATION_y)
        self.X_mesh, self.Y_mesh = np.meshgrid(self.x, self.y, indexing='xy')
        
        self.upper_bound = 1.5
        self.lower_bound = 1.0
        
        self.lambda1 = np.random.rand(self.N_DATA) * (self.upper_bound - self.lower_bound) + self.lower_bound
        self.lambda2 = np.random.rand(self.N_DATA) * (self.upper_bound - self.lower_bound) + self.lower_bound
        self.lambda3 = np.random.rand(self.N_DATA) * (self.upper_bound - self.lower_bound) + self.lower_bound

        # Ogden parameters for polybutadiene rubber (polyconvex). Moduli mu_p in
        # MPa. All mu > 0 and the largest exponent is 7.2 (vs the old 38.283),
        # so the stress stays O(1e2) instead of exploding to ~1e6 and the two
        # principal components P^xx / P^yy stay comparable in magnitude.
        self.mu = [630, 12, 4]           # mu1, mu2, mu3
        self.alpha = [1.30, 5.00, 7.20]  # alpha1, alpha2, alpha3
    
    def _create_axis(self, start, end, discretization):
        """
        Creates a discretized axis between two points.

        Args:
            start (float): Start of the axis.
            end (float): End of the axis.

        Returns:
            tuple: A tuple containing the discretized axis and the step size.
        """
        axis = np.linspace(start, end, discretization).astype(np.float64)
        step_size = (end - start) / (discretization - 1) 
        return axis, step_size

    def _expand_dims(self, lambda1, lambda2, lambda3, X, Y):
        """
        Expands the dimensions of input arrays for broadcasting in calculations.

        Args:
            lambda1 (array): First generator output.
            lambda2 (array): Second generator output.
            X (array): Meshgrid X coordinates.
            Y (array): Meshgrid Y coordinates.

        Returns:
            tuple: Expanded versions of lambda1, lambda2, X, and Y.
        """
        lambda1 = lambda1[:, np.newaxis, np.newaxis] 
        lambda2 = lambda2[:, np.newaxis, np.newaxis] 
        lambda3 = lambda3[:, np.newaxis, np.newaxis]
        
        X = X[np.newaxis, :, :] 
        Y = Y[np.newaxis, :, :]  

        return lambda1, lambda2, lambda3, X, Y
    
    def step_size(self):

        return self.x_step_size, self.y_step_size

    def Psi_func(self, lambda1, lambda2, lambda3, X, Y):
        """
        Computes the Ogden strain-energy density Psi from the principal stretches.

        Args:
            lambda1, lambda2, lambda3 (array): Principal stretches.
            X, Y (array): Meshgrid coordinates (used only for broadcasting shape).

        Returns:
            array: The computed Psi field.
        """
        lambda1, lambda2, lambda3, X, Y = self._expand_dims(lambda1, lambda2, lambda3, X, Y)

        Psi = np.sum([
            self.mu[i] / self.alpha[i] * (lambda1 ** self.alpha[i] + lambda2 ** self.alpha[i] + lambda3 ** self.alpha[i] - 3) for i in range(len(self.mu))
        ], axis=0)

        return Psi

    def U_func(self, lambda1, lambda2, lambda3, X, Y):
        """
        Computes the displacement field (Ux, Uy) of the homogeneous biaxial
        deformation defined by the principal stretches lambda1, lambda2.

        Args:
            lambda1, lambda2, lambda3 (array): Principal stretches.
            X, Y (array): Meshgrid coordinates.

        Returns:
            array: The stacked (Ux, Uy) displacement field.
        """
        lambda1, lambda2, lambda3, X, Y = self._expand_dims(lambda1, lambda2, lambda3, X, Y)

        Ux = lambda1*X - X
        Uy = lambda2*Y - Y

        return np.stack([Ux, Uy], axis=-1)
    
    def F_func(self, lambda1, lambda2, lambda3, X, Y):
        """
        Computes the diagonal components (Fxx, Fyy) of the deformation
        gradient F, equal to the principal stretches for this homogeneous
        biaxial deformation.

        Args:
            lambda1, lambda2, lambda3 (array): Principal stretches.
            X, Y (array): Meshgrid coordinates (used only for broadcasting shape).

        Returns:
            array: The stacked (Fxx, Fyy) field.
        """
        lambda1, lambda2, lambda3, X, Y = self._expand_dims(lambda1, lambda2, lambda3, X, Y)

        Fxx = lambda1 * np.ones_like(X)
        Fyy = lambda2 * np.ones_like(Y)

        return np.stack([Fxx, Fyy], axis=-1)
    
    def E_func(self, lambda1, lambda2, lambda3, X, Y):
        """
        Computes the diagonal components (Exx, Eyy) of the Green-Lagrange
        strain tensor, E = 1/2 (lambda^2 - 1) for this homogeneous biaxial
        deformation.

        Args:
            lambda1, lambda2, lambda3 (array): Principal stretches.
            X, Y (array): Meshgrid coordinates (used only for broadcasting shape).

        Returns:
            array: The stacked (Exx, Eyy) field.
        """
        lambda1, lambda2, lambda3, X, Y = self._expand_dims(lambda1, lambda2, lambda3, X, Y)

        Exx = 1/2 * (lambda1**2 * np.ones_like(X) - 1)
        Eyy = 1/2 * (lambda2**2 * np.ones_like(Y) - 1)

        return np.stack([Exx, Eyy], axis=-1)
    
    def S_func(self, lambda1, lambda2, lambda3, X, Y):
        """
        Computes the diagonal components (Sxx, Syy) of the second
        Piola-Kirchhoff stress predicted by the Ogden model.

        Args:
            lambda1, lambda2, lambda3 (array): Principal stretches.
            X, Y (array): Meshgrid coordinates (used only for broadcasting shape).

        Returns:
            array: The stacked (Sxx, Syy) field.
        """
        lambda1, lambda2, lambda3, X, Y = self._expand_dims(lambda1, lambda2, lambda3, X, Y)

        Sxx = np.sum([
            self.mu[i] * (lambda1 ** self.alpha[i] - (lambda1 * lambda2) ** -self.alpha[i]) for i in range(len(self.mu))
        ], axis=0) * np.ones_like(X)

        Syy = np.sum([
            self.mu[i] * (lambda2 ** self.alpha[i] - (lambda1 * lambda2) ** -self.alpha[i]) for i in range(len(self.mu))
        ], axis=0) * np.ones_like(Y)

        return np.stack([Sxx, Syy], axis=-1)
    
    def P_func(self, lambda1, lambda2, lambda3, X, Y):
        """
        Computes the diagonal components (Pxx, Pyy) of the first
        Piola-Kirchhoff stress, obtained from the second Piola-Kirchhoff
        stress via Pxx = Sxx / lambda1, Pyy = Syy / lambda2.

        Args:
            lambda1, lambda2, lambda3 (array): Principal stretches.
            X, Y (array): Meshgrid coordinates (used only for broadcasting shape).

        Returns:
            array: The stacked (Pxx, Pyy) field.
        """
        lambda1, lambda2, lambda3, X, Y = self._expand_dims(lambda1, lambda2, lambda3, X, Y)

        S = self.S_func(self.lambda1, self.lambda2, self.lambda3, self.X_mesh, self.Y_mesh)

        Pxx = 1/lambda1*S[..., 0]
        Pyy = 1/lambda2*S[..., 1]

        return np.stack([Pxx, Pyy], axis=-1)
    
    def P_tensor(self, lambda1, lambda2, lambda3, X, Y):
        """
        Assembles the full first Piola-Kirchhoff stress P^{ij} as a 2x2 field.

        P_func returns only the two principal (diagonal) components; here they are
        placed on the diagonal of a full 2x2 tensor field. The off-diagonal terms
        vanish for the diagonal biaxial deformation. This full tensor is what gets
        contracted with the boundary normal to form the reference traction.

        Returns:
            np.ndarray: P^{ij} of shape (N, 2, 2, Ny, Nx).
        """
        P_diag = self.P_func(lambda1, lambda2, lambda3, X, Y)  # (N, Ny, Nx, 2)
        N, Ny, Nx, _ = P_diag.shape
        P = np.zeros((N, 2, 2, Ny, Nx), dtype=P_diag.dtype)
        P[:, 0, 0] = P_diag[..., 0]
        P[:, 1, 1] = P_diag[..., 1]
        return P

    def T_func(self, lambda1, lambda2, lambda3, X, Y, normal):
        """
        Reference traction vector T^i = P^{ij} N_j for an outward normal N.

        Contracts the full first Piola-Kirchhoff stress with the reference
        boundary normal (as in the psi_pgnniv reference), so the traction is a
        genuine vector and its tangential (shear) component is exposed.

        Args:
            normal (tuple[float, float]): Outward unit normal (N_x, N_y) of the
                reference boundary face.

        Returns:
            np.ndarray: Traction field T^i of shape (N, 2, Ny, Nx).
        """
        P = self.P_tensor(lambda1, lambda2, lambda3, X, Y)  # (N, 2, 2, Ny, Nx)
        n = np.asarray(normal, dtype=P.dtype)
        # T^i = P^{ij} N_j  (sum over the reference index j).
        return np.einsum('bijmn,j->bimn', P, n)

    def boundary_tractions(self, lambda1, lambda2, lambda3, X, Y):
        """
        Full reference traction vectors on the positive x and y boundaries.

        The meshgrid layout (indexing='xy') is (Ny, Nx): axis -2 is y, axis -1 is
        x, so the positive faces x=xN and y=yN are the last index along each axis.

        Cell-averaged along the face direction so they land on the interior
        (Ny-1 / Nx-1) grid where the predicted stress lives, with the
        singleton axis of the fixed boundary coordinate reinserted.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - Tx: traction on the x-normal face N=(1,0) sampled at x=xN,
                  shape (N, 2, Ny-1, 1) -- components (P_xx, P_yx) along y.
                - Ty: traction on the y-normal face N=(0,1) sampled at y=yN,
                  shape (N, 2, 1, Nx-1) -- components (P_xy, P_yy) along x.
        """
        T_x = self.T_func(lambda1, lambda2, lambda3, X, Y, normal=(1.0, 0.0))
        Tx = T_x[:, :, :, -1]                       # x = xN -> (N, 2, Ny)
        Tx = 0.5 * (Tx[..., :-1] + Tx[..., 1:])     # (N, 2, Ny-1)
        Tx = Tx[..., np.newaxis]                    # (N, 2, Ny-1, 1)

        T_y = self.T_func(lambda1, lambda2, lambda3, X, Y, normal=(0.0, 1.0))
        Ty = T_y[:, :, -1, :]                       # y = yN -> (N, 2, Nx)
        Ty = 0.5 * (Ty[..., :-1] + Ty[..., 1:])     # (N, 2, Nx-1)
        Ty = Ty[:, :, np.newaxis, :]                # (N, 2, 1, Nx-1)

        return Tx, Ty

    def add_noise(self, data):
        """
        Adds uniform random noise to the input dataset.

        Args:
            data: numpy.ndarray
                Input data array with shape (N, H, W), where:
                - N is the number of samples.
                - H and W are the spatial dimensions.

        Returns:
            numpy.ndarray
                A new NumPy array of the same shape as `data`, but with added noise.
        """
        noise_added_data = np.max(data, axis=(1, 2))
        sigma = self.noise_sigma * noise_added_data
        
        # Generate random uniform noise with the computed sigma as bounds.
        noise_uniform = np.random.uniform(
            low = -sigma[:, None, None],  # Lower bound for each sample.
            high = sigma[:, None, None],  # Upper bound for each sample.
            size = data.shape             # Shape of noise array.
        )
        
        noise_added_data = data + noise_uniform
        
        return noise_added_data


    def generate_dataset(self, test_size=0.2):
        """
        Generates a synthetic dataset based on the defined functions and splits it into training and validation sets.

        Args:
            test_size (float, optional): Proportion of the dataset to include in the test split (default is 0.2).

        Returns:
            dict: A dictionary containing the training and validation datasets and their respective parameters.
        """
        # Calculate function outputs for u, qx, qy, and k
        U = self.U_func(self.lambda1, self.lambda2, self.lambda3, self.X_mesh, self.Y_mesh)
        F = self.F_func(self.lambda1, self.lambda2, self.lambda3, self.X_mesh, self.Y_mesh)
        E = self.E_func(self.lambda1, self.lambda2, self.lambda3, self.X_mesh, self.Y_mesh)
        S = self.S_func(self.lambda1, self.lambda2, self.lambda3, self.X_mesh, self.Y_mesh)
        P = self.P_func(self.lambda1, self.lambda2, self.lambda3, self.X_mesh, self.Y_mesh)
        Tx, Ty = self.boundary_tractions(self.lambda1, self.lambda2, self.lambda3, self.X_mesh, self.Y_mesh)

        # Create a dictionary to hold the datasets and parameters
        data_dict = {
            "N_DATA": self.N_DATA,
            "N_DISCRETIZATION_x": self.N_DISCRETIZATION_x,
            "N_DISCRETIZATION_y": self.N_DISCRETIZATION_y,
            "x0": self.x0, "xN": self.xN,
            "y0": self.y0, "yN": self.yN,
            "x": self.x, "y": self.y,
            "X_mesh": self.X_mesh, "Y_mesh": self.Y_mesh,
            "x_step_size": self.x_step_size, "y_step_size": self.y_step_size,
            "U": U,
            "F": F,
            "E": E,
            "S": S,
            "P": P,
            "Tx": Tx, "Ty": Ty,
            "lambda1": self.lambda1, "lambda2": self.lambda2, "lambda3": self.lambda3,
        }

        return data_dict
    
    def save_data(self, data_dict):
        """
        Saves the generated dataset to a file in pickle format.

        Args:
            data_dict (dict): The dictionary containing the dataset to be saved.
        """
        create_folder(self.PATH)

        base_pickle_path = os.path.join(self.PATH, f"{self.MODEL_NAME}.pkl")
        pickle_path = base_pickle_path 
        file_index = 1

        # Check if the file already exists and increment the file index if necessary
        while os.path.exists(pickle_path):
            file_index += 1
            pickle_path = os.path.join(self.PATH, f"{self.MODEL_NAME}_v{file_index}.pkl")
            
        with open(pickle_path, "wb") as f:
            pickle.dump(data_dict, f) 

        print(f"Data saved in {pickle_path}")