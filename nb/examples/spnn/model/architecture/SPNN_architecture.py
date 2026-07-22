import torch.nn as nn
import tensopsciml as to

from tensopsciml.tensors import TensOps

class SPNN(nn.Module):
    """Structure-Preserving Neural Network (SPNN).

    This model predicts tensor components (A and B) to evolve a state z 
    while maintaining physical constraints via tensor contractions.

    Attributes:
        dim_in: The number of input dimensions.
        dim_out: The total number of output neurons.
        model: The underlying sequential neural network.
    """

    def __init__(
        self, 
        n_inputs = 10, 
        n_hidden_layers = 3, 
        n_neurons = 20
    ):
        """Initializes SPNN with the given architecture."""
        super().__init__()

        self.dim_in = n_inputs
        self.dim_out = 2 * (n_inputs**2) # Two matrices (n_inputs x n_inputs)

        layers = []

        # Input layer
        layers.append(nn.Linear(n_inputs, n_neurons))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(n_neurons, self.dim_out))

        self.model = nn.Sequential(*layers)

    def forward(self, z_in_, L, M, dt):
        """Performs the forward pass using structure-preserving integration.

        Args:
            z_in_: Input state tensor of shape (batch, n_inputs).
            L: Poisson tensor L^{ij} (order [2, 0], antisymmetric).
            M: Friction tensor M^{ij} (order [2, 0], symmetric).
            dt: Time step size.

        Returns:
            A tuple of (z_next, deg_E, deg_S).
        """
        # Get raw network output
        raw_output = self.model(z_in_)
        
        # Split and reshape into A and B matrices (batch, dim_in, dim_in)
        split_idx = self.dim_in * self.dim_in
        A_ = raw_output[:, :split_idx].view(-1, self.dim_in, self.dim_in)
        B_ = raw_output[:, split_idx:].view(-1, self.dim_in, self.dim_in)

        # Wrap in TensOps. A and B are emitted as fully COVARIANT [0, 2] tensors,
        # so that A_{ij} z^j and B_{ij} z^j are directly the covector gradients
        # dE_i, dS_i.
        A = TensOps(A_.unsqueeze(-1),
            domain_dim=1, codomain_dim=[self.dim_in, self.dim_in],
            order=[0, 2], contravariant_indices=None, covariant_indices=[0, 1]
        )

        B = TensOps(
            B_.unsqueeze(-1), domain_dim=1,
            codomain_dim=[self.dim_in, self.dim_in],
            order=[0, 2], contravariant_indices=None, covariant_indices=[0, 1]
        )

        z_in = TensOps(
            z_in_.unsqueeze(-1), domain_dim=1,
            codomain_dim=[self.dim_in],
            order=[1, 0], contravariant_indices=[0], covariant_indices=None
        )

        # L and M carry the system's fixed geometric structure (batch size 1);
        # broadcast them to z_in's batch size before contracting against it.
        batch_size = z_in_.shape[0]
        if L.values.shape[0] != batch_size:
            L = TensOps(L.values.expand(batch_size, *L.values.shape[1:]),
                        domain_dim=L.domain, codomain_dim=L.codomain, order=L.order,
                        contravariant_indices=L.contravariant, covariant_indices=L.covariant)
        if M.values.shape[0] != batch_size:
            M = TensOps(M.values.expand(batch_size, *M.values.shape[1:]),
                        domain_dim=M.domain, codomain_dim=M.codomain, order=M.order,
                        contravariant_indices=M.contravariant, covariant_indices=M.covariant)

        # Physics-informed contractions. A and B are [0, 2] COVARIANT, so
        # contracting their second index against z^j yields the COVECTOR
        # gradients dE_i = A_{ij} z^j and dS_i = B_{ij} z^j directly.
        dE = to.contraction(A, z_in, [[None], [1]], [[0], [None]])
        dS = to.contraction(B, z_in, [[None], [1]], [[0], [None]])

        # Degeneracy conditions M^{ij} dE_j = 0 and L^{ij} dS_j = 0: the entropy
        # gradient is in the kernel of the Poisson bracket and the energy
        # gradient in the kernel of the friction bracket.
        deg_E = to.contraction(M, dE, [[1], [None]], [[None], [0]])
        deg_S = to.contraction(L, dS, [[1], [None]], [[None], [0]])

        # Update z^i(t+dt) = z^i(t) + dt * (L^{ij} dE_j + M^{ij} dS_j)
        term_l = to.contraction(L, dE, [[1], [None]], [[None], [0]])
        term_m = to.contraction(M, dS, [[1], [None]], [[None], [0]])

        z_next = z_in + dt * (term_l + term_m)

        return z_next, deg_E, deg_S