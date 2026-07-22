import torch

from tensopsciml.tensors import TensOps

def get_train_val_indices(total_size, train_split, gen):
    """
    Generates consistent training and validation indices.

    Parameters:
        total_size (int): Total number of samples in the dataset.
        train_split (float): The proportion of the dataset to include in the training set.
        gen (torch.Generator): The generator for reproducibility.

    Returns:
        train_indices (Tensor): Indices for the training set.
        val_indices (Tensor): Indices for the validation set.
    """
    indices = torch.randperm(total_size, generator=gen)
    
    # Calculate the split point
    train_size = int(total_size * train_split)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    return train_indices, val_indices

def train_val_split_torch_tensops(X, train_indices, val_indices):
    """
    Splits a TensOps object into training and validation sets using precomputed indices.

    Parameters:
        X (TensOps/Tensor): The TensOps or Torch object to be split.
        train_indices (Tensor): Indices to be used for the training set.
        val_indices (Tensor): Indices to be used for the validation set.

    Returns:
        X_train (TensOps/Tensor): The training set.
        X_val (TensOps/Tensor): The validation set.

    Raises:
        ValueError: If the input is not a TensOps object or a Torch tensor.
    """
    if isinstance(X, TensOps):

        # Indices come from a CPU generator; move them to the data's device so
        # index_select does not mix CPU and CUDA tensors.
        train_indices = train_indices.to(X.values.device)
        val_indices = val_indices.to(X.values.device)

        # Extract values using the provided indices
        train_vals = torch.index_select(X.values, 0, train_indices)
        val_vals = torch.index_select(X.values, 0, val_indices)

        # Create TensOps objects for the training and validation sets
        X_train = TensOps(train_vals, domain_dim=X.domain, codomain_dim=X.codomain, order=X.order, contravariant_indices=X.contravariant, covariant_indices=X.covariant)
        X_val = TensOps(val_vals, domain_dim=X.domain, codomain_dim=X.codomain, order=X.order, contravariant_indices=X.contravariant, covariant_indices=X.covariant)

        return X_train, X_val

    elif isinstance(X, torch.Tensor):

        # Split the tensor using the provided indices (on X's device).
        X_train = X[train_indices.to(X.device)]
        X_val = X[val_indices.to(X.device)]

        return X_train, X_val
    
    else:
        raise ValueError("Object must be a Torch or a TensOps tensor.")