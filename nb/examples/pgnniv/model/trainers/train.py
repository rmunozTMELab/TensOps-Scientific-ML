import time 

from tensopsciml.tensors import TensOps
from model.utils.checkpoints import load_checkpoint, save_checkpoint
from .eval import loss_function

def train_epoch(model, optimizer, Tx_train, Ty_train, U_train, D):
    """
    Trains the model for one epoch.

    Parameters:
        model (torch.Model): The model being trained.
        optimizer (torch.Optimizer): The optimization algorithm used to update model parameters.
        Tx_train (torch.Tensor): Boundary-condition input on the x-face.
        Ty_train (torch.Tensor): Boundary-condition input on the y-face.
        U_train (TensOps): Ground-truth displacement field.
        D (torch.Tensor): Derivative filters.

    Returns:
        loss (float): Total computed loss for the training epoch.
        e_loss, pi1_loss, pi2_loss, pi3_loss (float): Specific components of the loss related to the model's performance (regularizations).
    """

    # Set the model to training mode and activate forward pass.
    model.train()

    U_pred, F_pred, E_pred, P_pred, Tx_pred, Ty_pred = model(Tx_train, Ty_train)

    # Compute the total loss and its components by passing the necessary arguments to the loss function.
    total_loss, MSE_R1, MSE_R2x, MSE_R2y, MSE_R3, MSE_R4x, MSE_R4y = loss_function(U_train, U_pred, P_pred, Tx_train, Ty_train, D)

    # Reset the gradients, perform backpropagation, and update model's parameters using optimizer and compute gradients.
    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    optimizer.step()

    # Return the total loss and its components.
    return total_loss, MSE_R1, (MSE_R2x + MSE_R2y), MSE_R3, (MSE_R4x + MSE_R4y)


def test_epoch(model, Tx_test, Ty_test, U_test, D):
    """
    Evaluates the model on the test dataset for one epoch.

    Parameters:
        model (torch.Model): The model being trained.
        optimizer (torch.Optimizer): The optimization algorithm used to update model parameters.
        Tx_test (torch.Tensor): Boundary-condition input on the x-face.
        Ty_test (torch.Tensor): Boundary-condition input on the y-face.
        U_test (TensOps): Ground-truth displacement field.
        D (torch.Tensor): Derivative filters.

    Returns:
        loss (float): Total computed loss for the training epoch.
        e_loss, pi1_loss, pi2_loss, pi3_loss (float): Specific components of the loss related to the model's performance (regularizations).
    """

    # Perform a forward pass with the test data to generate predictions
    U_pred, F_pred, E_pred, P_pred, Tx_pred, Ty_pred = model(Tx_test, Ty_test)

    # Compute the total loss and its components for the test data.
    total_loss, MSE_R1, MSE_R2x, MSE_R2y, MSE_R3, MSE_R4x, MSE_R4y = loss_function(U_test, U_pred, P_pred, Tx_test, Ty_test, D)

    # Return the total loss and its components.
    return total_loss, MSE_R1, (MSE_R2x + MSE_R2y), MSE_R3, (MSE_R4x + MSE_R4y)

def train_loop(model, optimizer, 
               Tx_train, Ty_train, U_train, Tx_test, Ty_test, U_test, 
               D, n_checkpoints, start_epoch, n_epochs, batch_size, 
               model_results_path, device, new_lr=None):
    """
    Main training loop for the model over multiple epochs.

    Parameters:
        model (torch.Model): The machine learning model to train.
        optimizer (torch.Optimizer): Optimizer used to update model parameters.
        n_checkpoints (int): Number of checkpoints to save during training.

        Tx_train (torch.Tensor): Training boundary-condition input on the x-face.
        Ty_train (torch.Tensor): Training boundary-condition input on the y-face.
        U_train (TensOps): Ground-truth training displacement field.
        Tx_test (torch.Tensor): Test boundary-condition input on the x-face.
        Ty_test (torch.Tensor): Test boundary-condition input on the y-face.
        U_test (TensOps): Ground-truth test displacement field.
        D (torch.Tensor): Derivative filters.

        start_epoch (int): Epoch to start training from.
        n_epochs (int): Total number of epochs to train the model.
        batch_size (int): Number of samples per training batch.
        model_results_path (str): Path to save model checkpoints and results.
        device (str): Device to use for computation (e.g., 'cpu' or 'cuda').
        new_lr (int): Optional. New learning rate to set if resuming from a checkpoint.
    """
    # Handle case where training resumes from a saved checkpoint
    if start_epoch > 0:
        # Load the saved model, optimizer, and associated lists from checkpoint
        print(f'Starting training from a checkpoint. Epoch {start_epoch}.')
        resume_epoch = start_epoch
        model, optimizer, lists = load_checkpoint(model, optimizer, resume_epoch, model_results_path)

        # Update learning rate if a new one is specified
        if new_lr != None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        time_list = lists['time_list']

        train_total_loss_list = lists['train_total_loss_list']
        train_R1_loss_list = lists['train_R1_loss_list']
        train_R2_loss_list = lists['train_R2_loss_list']
        train_R3_loss_list = lists['train_R3_loss_list']
        train_R4_loss_list = lists['train_R4_loss_list']

        test_total_loss_list = lists['test_total_loss_list']
        test_R1_loss_list = lists['test_R1_loss_list']
        test_R2_loss_list = lists['test_R2_loss_list']
        test_R3_loss_list = lists['test_R3_loss_list']
        test_R4_loss_list = lists['test_R4_loss_list']

    # Handle case where training starts from scratch
    else:
        print("Starting training from scratch.")

        train_total_loss_list = []
        train_R1_loss_list = []
        train_R2_loss_list = []
        train_R3_loss_list = []
        train_R4_loss_list = []

        test_total_loss_list = []
        test_R1_loss_list = []
        test_R2_loss_list = []
        test_R3_loss_list = []
        test_R4_loss_list = []

        time_list = []
    
    # Get the number of samples in training and test datasets
    N_train = Tx_train.shape[0]
    N_test = Tx_test.shape[0]

    # The test set is evaluated whole every epoch, so move it to the compute
    # device once here (training batches are moved to device inside the loop).
    Tx_test = Tx_test.to(device)
    Ty_test = Ty_test.to(device)
    U_test = TensOps(U_test.values.to(device),
                     domain_dim=U_test.domain,
                     codomain_dim=U_test.codomain,
                     order=U_test.order,
                     contravariant_indices=U_test.contravariant,
                     covariant_indices=U_test.covariant)

    # Epochs between checkpoints.
    checkpoint_every = max(1, int(n_epochs / n_checkpoints))

    if start_epoch >= n_epochs:
        print(f"Nothing to train: start_epoch ({start_epoch}) >= n_epochs ({n_epochs}).")
        return

    # Start training loop for specified number of epochs
    for epoch_i in range(start_epoch, n_epochs):
        
        # Time instant when starts the training of the model for one epoch
        time_init = time.time()

        # Process training data in batches and prepare them in the needed format
        for batch_start in range(0, N_train, batch_size):
            Tx_batch = Tx_train[batch_start:(batch_start+batch_size)].to(device)
            Ty_batch = Ty_train[batch_start:(batch_start+batch_size)].to(device)
            U_batch = TensOps(U_train.values[batch_start:(batch_start+batch_size)].to(device), 
                              domain_dim=U_train.domain, 
                              codomain_dim=U_train.codomain, 
                              order=U_train.order, 
                              contravariant_indices=U_train.contravariant, 
                              covariant_indices=U_train.covariant)
            
            train_loss, R1_loss, R2_loss, R3_loss, R4_loss = train_epoch(model, optimizer, Tx_batch, Ty_batch, U_batch, D)


        # Time instant when ends  the training of the model for one epoch and obtaining the total time
        time_end = time.time()
        train_time_epoch = time_end - time_init

        # Record average training losses after processing all batches
        time_list.append(train_time_epoch)
        train_total_loss_list.append(train_loss.item())
        train_R1_loss_list.append(R1_loss.item())
        train_R2_loss_list.append(R2_loss.item())
        train_R3_loss_list.append(R3_loss.item())
        train_R4_loss_list.append(R4_loss.item())

        # Evaluate model on the test dataset at the end of each epoch
        loss_test, R1_test, R2_test, R3_test, R4_test = test_epoch(model, Tx_test, Ty_test, U_test, D)

        # Record average test losses
        test_total_loss_list.append(loss_test.item())
        test_R1_loss_list.append(R1_test.item())
        test_R2_loss_list.append(R2_test.item())
        test_R3_loss_list.append(R3_test.item())
        test_R4_loss_list.append(R4_test.item())

        # Print training and testing metrics periodically
        if epoch_i % (1 if n_epochs < 100 else (10 if n_epochs <= 1000 else 100)) == 0:
            print(f'Epoch {epoch_i}, Train loss: {train_loss.item():.3e}, Test loss: {loss_test.item():.3e}, MSE(R1): {R1_loss.item():.3e}, MSE(R2): {R2_loss.item():.3e}, MSE(R3): {R3_loss.item():.3e}, MSE(R4): {R4_loss.item():.3e}')

        # Save a checkpoint at regular intervals
        if epoch_i % checkpoint_every == 0:
            save_checkpoint(model, optimizer, epoch_i, model_results_path, time_list=time_list,
                            train_total_loss_list=train_total_loss_list, test_total_loss_list=test_total_loss_list,
                            train_R1_loss_list=train_R1_loss_list, train_R2_loss_list=train_R2_loss_list, 
                            train_R3_loss_list=train_R3_loss_list, train_R4_loss_list=train_R4_loss_list,
                            test_R1_loss_list=test_R1_loss_list, test_R2_loss_list=test_R2_loss_list, 
                            test_R3_loss_list=test_R3_loss_list, test_R4_loss_list=test_R4_loss_list)

    save_checkpoint(model, optimizer, epoch_i, model_results_path, time_list=time_list,
                    train_total_loss_list=train_total_loss_list, test_total_loss_list=test_total_loss_list,
                    train_R1_loss_list=train_R1_loss_list, train_R2_loss_list=train_R2_loss_list, 
                    train_R3_loss_list=train_R3_loss_list, train_R4_loss_list=train_R4_loss_list,
                    test_R1_loss_list=test_R1_loss_list, test_R2_loss_list=test_R2_loss_list, 
                    test_R3_loss_list=test_R3_loss_list, test_R4_loss_list=test_R4_loss_list)
    
    print("\nTraining process finished after", n_epochs, "epochs\n")
    
    save_checkpoint(model, optimizer, epoch_i, model_results_path, time_list=time_list,
                    train_total_loss_list=train_total_loss_list, test_total_loss_list=test_total_loss_list,
                    train_R1_loss_list=train_R1_loss_list, train_R2_loss_list=train_R2_loss_list, 
                    train_R3_loss_list=train_R3_loss_list, train_R4_loss_list=train_R4_loss_list,
                    test_R1_loss_list=test_R1_loss_list, test_R2_loss_list=test_R2_loss_list, 
                    test_R3_loss_list=test_R3_loss_list, test_R4_loss_list=test_R4_loss_list,
                    end_flag=True)