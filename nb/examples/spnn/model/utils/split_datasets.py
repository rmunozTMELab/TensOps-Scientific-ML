import numpy as np

def split_dataset(p: float, total_trajectories: int):
    """Splits trajectory indices into training and testing sets.

    Args:
        p: The proportion of the dataset to include in the train split (0.0 to 1.0).
        total_trajectories: The total number of trajectories available.

    Returns:
        A tuple containing two lists:
            - train_indices: A list of indices for the training set.
            - test_indices: A list of indices for the test set.
    """
    # Calculate the number of training samples
    num_train = int(p * total_trajectories)

    # Create a shuffled list of all indices
    indices = np.arange(total_trajectories)
    np.random.shuffle(indices)

    # Slice indices into train and test sets
    train_indices = indices[:num_train].tolist()
    test_indices = indices[num_train:].tolist()

    return train_indices, test_indices