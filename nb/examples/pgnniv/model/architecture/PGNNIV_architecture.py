import torch
import torch.nn as nn
import tensopsciml as to

from tensopsciml import TensOps
from model.utils.tractions import compute_boundary_tractions

class PredictiveEncoder(nn.Module):

    def __init__(self, input_size_1, input_size_2, hidden_layer_1_size, hidden_layer_2_size, latent_space_size):
        super(PredictiveEncoder, self).__init__()

        # Parameters
        self.in_size = torch.prod(torch.tensor(input_size_1)) + torch.prod(torch.tensor(input_size_2))
        self.h1_size = hidden_layer_1_size
        self.h2_size = hidden_layer_2_size
        self.ls_size = latent_space_size

        # Architecture
        self.flatten_layer = nn.Flatten(start_dim=1)  
        self.hidden1_layer = nn.Linear(self.in_size.item(), self.h1_size)
        self.hidden2_layer = nn.Linear(self.h1_size, self.h2_size)
        self.latent_space_layer = nn.Linear(self.h2_size, self.ls_size)
    
    def forward(self, X1, X2):
        dtype = X1.dtype  # Use the dtype of the input tensors
        X1 = X1.to(dtype=dtype)
        X2 = X2.to(dtype=dtype)

        X1_flat = self.flatten_layer(X1)
        X2_flat = self.flatten_layer(X2)
        X = torch.cat((X1_flat, X2_flat), dim=1)        
        X = torch.sigmoid(self.hidden1_layer(X))
        X = torch.sigmoid(self.hidden2_layer(X))
        latent_space_output = self.latent_space_layer(X)

        return latent_space_output

class PredictiveDecoder(nn.Module):

    def __init__(self, latent_space_size, hidden_layer_1_size, hidden_layer_2_size, output_size):
        super(PredictiveDecoder, self).__init__()

        # Parameters
        self.ls_size = latent_space_size
        self.h1_size = hidden_layer_1_size
        self.h2_size = hidden_layer_2_size
        self.out_size = torch.tensor(output_size)

        # Architecture
        self.hidden1_layer = nn.Linear(self.ls_size, self.h1_size)
        self.hidden2_layer = nn.Linear(self.h1_size, self.h2_size)
        self.output_layer = nn.Linear(self.h2_size, torch.prod(self.out_size).item())
    
    def forward(self, X):
        dtype = X.dtype  # Use the dtype of the input tensor
        X = X.to(dtype=dtype)
        
        X = torch.sigmoid(self.hidden1_layer(X))
        X = torch.sigmoid(self.hidden2_layer(X))
        decoder_output = self.output_layer(X)

        return decoder_output.view(decoder_output.size(0), *self.out_size.int().tolist())
    
class ExplanatoryNetwork(nn.Module):

    def __init__(self, input_size, n_filters, hidden_layer_size, output_size):
        super(ExplanatoryNetwork, self).__init__()

        # Parameters
        self.in_size = torch.tensor(input_size)
        self.n_filters = n_filters * 4
        self.h_layer = hidden_layer_size
        self.out_size = torch.tensor(output_size)

        # Architecture
        self.conv_expand_layer = nn.Conv2d(in_channels=4, out_channels=self.n_filters, kernel_size=1)
        self.hidden_layer_1 = nn.Linear(self.n_filters, self.h_layer)
        self.hidden_layer_2 = nn.Linear(self.h_layer, self.n_filters)
        self.conv_converge_layer = nn.Conv2d(in_channels=self.n_filters, out_channels=4, kernel_size=1)
        
    def forward(self, X):
        dtype = X.dtype  # Use the dtype of the input tensor
        X = X.to(dtype=dtype)

        batch_size, _, height, width = X.shape
        
        X = self.conv_expand_layer(X)  # (batch, channels, height, width)
        X = X.permute(0, 2, 3, 1)      # (batch, height, width, channels)
        X = X.reshape(-1, X.size(-1))  # (batch * height * width, channels)

        X = torch.sigmoid(self.hidden_layer_1(X))
        X = torch.sigmoid(self.hidden_layer_2(X))
        
        X = X.reshape(batch_size, height, width, -1)
        X = X.permute(0, 3, 1, 2)  
        explanatory_output = self.conv_converge_layer(X)

        return explanatory_output
    
class PGNNIV(nn.Module):
    
    def __init__(self, input_size_1, input_size_2, predictive_layers, output_predictive_size, explanatory_input_size, explanatory_layers, output_explanatory_size, n_filters, D_filter, metric, input_scale=1.0, output_scale=1.0):
        super(PGNNIV, self).__init__()

        # Parameters
        self.in1_size = input_size_1
        self.in2_size = input_size_2
        self.pred_size = predictive_layers
        self.out_pred_size = output_predictive_size

        self.in_exp_size = explanatory_input_size
        self.exp_size = explanatory_layers
        self.out_exp_size = output_explanatory_size

        self.n_filters = n_filters
        self.D_filter = D_filter

        # Metric of the problem: a batch-1 MetrOps on the grid where GradU
        # lives. Supplied by the caller instead of being fabricated here, so
        # the architecture stays agnostic to the choice of metric. forward()
        # broadcasts it to whatever batch size the current step uses.
        self.metric = metric

        # Characteristic scale of the boundary-condition inputs (Tx, Ty).
        self.register_buffer("input_scale", torch.as_tensor(float(input_scale)))

        # Characteristic scale of the stress output. 
        self.register_buffer("output_scale", torch.as_tensor(float(output_scale)))

        # Architecture
        self.encoder = PredictiveEncoder(self.in1_size, self.in2_size, self.pred_size[0], self.pred_size[1], self.pred_size[2])
        self.decoder = PredictiveDecoder(self.pred_size[2], self.pred_size[3], self.pred_size[4], self.out_pred_size)
        self.explanatory = ExplanatoryNetwork(self.in_exp_size, self.n_filters, *self.exp_size, self.out_exp_size)

    def forward(self, X1, X2):

        # Align the inputs with the model's own device, so a direct forward call
        # (e.g. from a visualizer notebook) works whether the boundary
        # conditions were built on CPU or GPU.
        device = next(self.parameters()).device
        X1 = X1.to(device)
        X2 = X2.to(device)

        # Non-dimensionalize the boundary-condition inputs so the encoder's
        # sigmoids do not saturate.
        X1 = X1 / self.input_scale
        X2 = X2 / self.input_scale

        # Predictive network
        X = self.encoder(X1, X2)
        X = self.decoder(X)
        
        U_ = X.view(X.size(0), *self.out_pred_size)
        U = TensOps(U_, domain_dim=2, codomain_dim=[2], order=[1, 0], contravariant_indices=[0], covariant_indices=None)  # Displacement

        # The derivative kernels are a plain attribute, so model.to(device) does
        # not move them; align them with the input's device (cached after the
        # first forward) so the convolutions do not mix CPU and CUDA tensors.
        self.D_filter = self.D_filter.to(U.values.device)
        GradU = to.partial_derivative(U, self.D_filter)  # d_j U^i (covariant derivative index)

        # Metric of the problem, supplied at construction time (see __init__)
        # as a batch-1 MetrOps on GradU's grid. Broadcast it to the running
        # batch size -- contraction() needs both operands to share the exact
        # same batch -- and align it with the running device, as D_filter is
        # realigned just above.
        N = GradU.values.shape[0]
        metric_ = self.metric.values.to(GradU.values.device).expand(N, *self.metric.values.shape[1:])
        metric = to.MetrOps(metric_, domain_dim=self.metric.domain, codomain_dim=self.metric.codomain)

        I = to.kronecker_delta(metric)  # mixed identity delta^i_j

        F = GradU + I  # Deformation tensor

        # C = F^T g F (right Cauchy-Green)
        F_low = to.lower_index(metric, F)                           # F_ik
        C = to.contraction(F_low, F, [[None], [0]], [[0], [None]])  # C_kl [0, 2]

        # Green-Lagrange strain E_kl = 1/2 (C_kl - g_kl).
        E = (C - to.lower_index(metric, I))/2  # Green-Lagrange tensor

        # Explanatory network.
        E_ = E.values.reshape(E.values.size(0), -1, *E.values.shape[-2:])
        S_ = self.explanatory(E_) * self.output_scale
        S = TensOps(S_.reshape(S_.size(0), *self.out_exp_size), domain_dim=2, codomain_dim=[2, 2], order=[2, 0], contravariant_indices=[0, 1], covariant_indices=None)

        # S is the stress conjugate to the symmetric C, and ITS SYMMETRY is what
        # enforces the balance of angular momentum. The network's four output
        # channels are independent, so symmetrize explicitly.
        S = to.symmetrize(S, [0, 1], 'contravariant')  # S^{kl} = S^{lk}

        # First Piola-Kirchhoff stress P^{ij} = F^i_k S^{kj}.
        P = to.contraction(F, S, [[None], [0]], [[0], [None]])  # P^{ij} [2, 0]

        # Boundary tractions T^i = P^{ij} N_j on the positive x and y faces (the
        # same operator R3 uses), returned as full vectors.
        tx_pred, ty_pred = compute_boundary_tractions(P)

        return U, F, E, P, tx_pred.values, ty_pred.values