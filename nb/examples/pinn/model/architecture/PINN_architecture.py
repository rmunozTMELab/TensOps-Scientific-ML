import torch.nn as nn

class PINN(nn.Module):
    
    def __init__(self, n_inputs=1, n_outputs=1, n_hidden_layers=3, n_neurons=10):
        super(PINN, self).__init__()

        layers = []
        
        # Input layer
        layers.append(nn.Linear(n_inputs, n_neurons))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(n_neurons, n_outputs))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)