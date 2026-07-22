# TensOpSciML: A Python-Based Framework for Tensor Calculus and Differential Operators with Application in Scientific Machine Learning.

<p align="center">
  <a href="https://github.com/rmunozTMELab/TensOps-Scientific-ML/stargazers">
    <img src="https://img.shields.io/github/stars/rmunozTMELab/TensOps-Scientific-ML?style=social" alt="Stars">
  </a>
  <a href="https://github.com/rmunozTMELab/TensOps-Scientific-ML/network/members">
    <img src="https://img.shields.io/github/forks/rmunozTMELab/TensOps-Scientific-ML?style=social" alt="Forks">
  </a>
  <a href="https://github.com/rmunozTMELab/TensOps-Scientific-ML/watchers">
    <img src="https://img.shields.io/github/watchers/rmunozTMELab/TensOps-Scientific-ML?style=social" alt="Watchers">
  </a>
</p>

<p align="center">
  <a href="https://github.com/rmunozTMELab/TensOps-Scientific-ML/issues">
    <img src="https://img.shields.io/github/issues/rmunozTMELab/TensOps-Scientific-ML" alt="Issues">
  </a>
  <a href="https://github.com/rmunozTMELab/TensOps-Scientific-ML/pulls">
    <img src="https://img.shields.io/github/issues-pr/rmunozTMELab/TensOps-Scientific-ML" alt="Pull Requests">
  </a>
  <a href="https://github.com/rmunozTMELab/TensOps-Scientific-ML/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/rmunozTMELab/TensOps-Scientific-ML" alt="License">
  </a>
  <a href="https://github.com/rmunozTMELab/TensOps-Scientific-ML/commits/main">
    <img src="https://img.shields.io/github/last-commit/rmunozTMELab/TensOps-Scientific-ML" alt="Last Commit">
  </a>
</p>

## Abstract
Scientific Machine Learning (SciML) seeks to integrate physical principles into data-driven models to achieve robust, generalizable and interpretable results. These principles are typically encoded through rigorous mathematical formulations involving differential operators, tensor fields, and high-order multilinear structures. Ensuring that such operations remain compatible with Deep Learning (DL) frameworks, and therefore with automatic differentiation and modern GPU architectures, presents a significant challenge. In this work, we introduce `TensOpSciML`, a unified Python framework designed for tensor calculus and differential operators for data-driven applications. `TensOpSciML` provides two core contributions: first, it offers a comprehensive suite of convolution-based derivative operators, equivalent to finite-difference scheme, that enables accurate, efficient and fully differentiable approximations of spatial derivatives in one, two and three dimensions. Secondly, the framework enriches scientific-computing tensors with explicit mathematical metadata, such as domain dimension, tensorial order, and covariant and contravariant indices, thus elevating them from raw numerical arrays to well-structured mathematical objects. This additional context enables `TensOpSciML` to support a broad class of algebraic and differential operations (e.g., tensor contractions, outer and inner products, gradient, divergence, curl), all generalized to n-dimensional Euclidean spaces. Although it is implemented in PyTorch, `TensOpSciML` is backend-agnostic: all operations can be implemented in any of the most widely used scientific computing frameworks in Python, as they are TensorFlow (TF) or JAX. The aim of `TensOpSciML` is to unify rigorous mathematical foundations with modern Machine Learning, guaranteeing  that every computation respects the full mathematical structure of the tensorial objects involved, thereby maintaining correctness, interpretability, and mathematical rigor in SciML contexts.



## 🎯 Features

### Core Tensor Operations
- **Tensor Abstraction**: Universal tensor representation via the `TensOps` class: it represents domain, codomain, order, and contravariant and covariant indices.
- **Backend Flexibility**: PyTorch backend with pluggable architecture for extensibility.
- **Device Agnostic**: Seamless CPU/GPU support through backend abstraction.

### Finite Difference Kernels
- **Derivative Kernels**: First-order finite difference kernels for 1D, 2D, and 3D.
- **Averaging Kernels**: Smoothing kernels for numerical averaging in multiple dimensions.
- **Flexible Configuration**: Customizable grid spacing, indexing conventions, and data types.

### Tensor Algebra
- **Index Operations**: Full support for index raising/lowering (via an explicit metric), contractions, and transformations.
- **Invariants**: Computation of first, second, and third tensor invariants.
- **Matrix Operations**: Determinant, inverse, transpose, trace, and spherical/deviatoric splits.
- **Advanced Operations**:
  - Tensor (outer) product.
  - Contraction and self-contraction.
  - Symmetrization and antisymmetrization (covariant and contravariant).
  - Cross product.
  - Levi-Civita symbol and (metric-weighted) tensor computations.

### Geometry & Metrics
- **Metric Tensor**: `MetrOps` wraps a symmetric positive-definite metric field, with validation and index-raising/lowering support.
- **Volume Form**: `VolFormOps` derives the volume form from a metric for divergence and curl.
- **Connections**: `ConnectOps` builds the Levi-Civita connection from a metric, enabling covariant differentiation on non-Cartesian geometries.
- **No fabricated defaults**: metrics, inverse metrics, volume forms, and connections are always passed explicitly — nothing Euclidean is assumed silently.

### Calculus Operations
- **Partial Derivatives**: `partial_derivative` (the ∂ operator, adding a covariant index) and the directional kernels `d1`, `d2`, `d3`.
- **Differential Operators**: Metric-aware `gradient` (scalar → contravariant vector), `divergence`, `curl`, and `laplacian`.
- **Covariant Operators**: `cov_derivative`, `cov_divergence`, `cov_curl`, `cov_directional_derivative`, and `lie_derivative`, all driven by a `ConnectOps` connection.
- **Advanced Methods**: Advection operator and automatic differentiation (`autograd`) support.

All these operations proceed from a mathematically oriented standpoint, emphasizing rigor, consistency, and strict adherence to tensor-index notation.


## 📚 Project Structure

```
tensopsciml-project/
├── src/
│   └── tensopsciml/
│       ├── algebra/          # invariants, matrix_algebra, tensor_algebra
│       ├── backends/         # backend registry + PyTorch backend
│       ├── calculus/         # operators, cov_operators, connections
│       ├── kernels/          # derivative and averaging kernels
│       ├── tensors/          # TensOps, TensorConfig, metrics (MetrOps/VolFormOps)
│       └── utils/            # indices, averaging, Levi-Civita symbol, validation
├── tests/
│   ├── algebra/
│   ├── backends/
│   ├── calculus/
│   ├── data/
│   ├── kernels/
│   ├── tensors/
│   └── utils/
├── nb/
│   ├── examples/
│   │   ├── pgnniv/
│   │   ├── pinn/
│   │   └── spnn/
│   └── tutorials/
│       ├── data/
│       ├── tensops_tutorial.ipynb
│       ├── tensorial_identities.ipynb
│       ├── operators_validation.ipynb
│       └── cov_operators_validation.ipynb
├── MANIFEST.in
├── README.md
└── pyproject.toml
```


## 📦 Installation

### Prerequisites
- Python 3.11+
- PyTorch 2.1+
- NumPy 1.24+
- Sympy 1.12+
- Scipy 1.10+

### Using pip

```bash
python -m pip install tensopsciml
```

---

## 🚀 Quick Start

### Basic Tensor Creation

```python
import torch
from tensopsciml import TensOps

# Create a tensor with contravariant indices (upper) and covariant indices (lower)
# It represents a 3x3 matrix (A^i_j), with a three-dimensional discretization (10x10x10).

data = torch.randn(1, 3, 3, 10, 10, 10)

tensor = TensOps(
  data,
  domain_dim=3,
  codomain_dim=[3, 3],
  order=[1, 1],
  contravariant_indices=[0],
  covariant_indices=[1],
  )

print(tensor)

# <class 'TensOps'>
#  Shape: [1, 3, 3, 10, 10, 10]

#  Order: [1, 1]
#  Domain dim: 3
#  Codomain dim: [3, 3]

#  Base: V x V*
#  Contravariant indices: [0]
#  Covariant indices: [1]
```

### Using Differential Operators

```python
import torch
from tensopsciml import TensOps, Derivative, partial_derivative

# Derivative kernels matching the tensor domain
derivative = Derivative(d1=0.1, d2=0.1)
D = derivative.two_dimensions()


vector_values = torch.rand((8, 2, 32, 32))
vector = TensOps(
  vector_values,
  domain_dim=2,
  codomain_dim=[2],
  order=[1, 0],
  contravariant_indices=[0],
  covariant_indices=None,
)

# ``partial_derivative`` (the ∂ operator) appends a covariant index.
# Note: differentiating shrinks the spatial grid by one node per direction.
grad_vector = partial_derivative(vector, D)
```

### Metric-Aware Operators

Every metric-dependent operator (`gradient`, `divergence`, `curl`, `laplacian`)
requires an explicit metric — nothing Euclidean is assumed silently.

```python
import torch
from tensopsciml import (
    TensOps, Derivative, MetrOps, VolFormOps,
    inverse, gradient, divergence,
)

D = Derivative(d1=0.1, d2=0.1, d3=0.1).three_dimensions()

# Build the Euclidean metric g_ij = δ_ij on a 3D grid, plus its inverse
# and the derived volume form.
metric_values = torch.eye(3).reshape(1, 3, 3, 1, 1, 1).repeat(1, 1, 1, 10, 10, 10)
metric = MetrOps(metric_values, domain_dim=3, codomain_dim=[3, 3])
inv_metric = inverse(metric)
volume_form = VolFormOps(metric)

# Scalar field  ->  gradient raises to a contravariant vector.
scalar = TensOps(
    torch.rand(1, 1, 10, 10, 10),
    domain_dim=3, codomain_dim=[1], order=[0, 0],
    contravariant_indices=None, covariant_indices=None,
)
grad = gradient(scalar, inv_metric, D)
```

### Tensor Algebra Operations

```python
from tensopsciml.algebra import tensor_product, contraction, first_invariant

# A and B are matrices (second-order tensor).
data = torch.randn(1, 3, 3, 10, 10, 10)

A = TensOps(
  data,
  domain_dim=3,
  codomain_dim=[3, 3],
  order=[1, 1],
  contravariant_indices=[0],
  covariant_indices=[1],
  )

B = TensOps(
  data,
  domain_dim=3,
  codomain_dim=[3, 3],
  order=[1, 1],
  contravariant_indices=[0],
  covariant_indices=[1],
  )

C = tensor_product(A, B)
D = contraction(A, B, [[0], [None]], [[None], [0]])
I1 = first_invariant(A)
```



## 📖 Tutorials

The `nb/tutorials/` directory contains interactive Jupyter notebooks designed to help you learn TensOps:

### 1. *tensops_tutorial.ipynb*
  Builds intuition for the `TensOps` wrapper by sampling symbolic functions on 1D/2D/3D grids and lifting them to:
  - Scalar fields with order `[0, 0]`
  - Vector fields with contravariant indices
  - Second- and third-order tensor fields with explicit domain/codomain metadata
  - Summaries that relate mathematical notation to PyTorch tensor shapes

### 2. *tensorial_identities.ipynb*
  Recreates some algebraic expressions from *Holzapfel, G. A. (2002). Nonlinear solid mechanics: a continuum approach for engineering science.* by testing:
  - Vector identities (dot products, triple products, dyadic products)
  - Tensor identities covering matrix-vector actions, dyads, and triple contractions
  - Determinant, transpose, and (anti-)symmetrization properties
  - Numerical proofs that the implemented operations respect the theoretical equalities

### 3. *operators_validation.ipynb*
  Verifies vector-calculus identities numerically using finite-difference kernels and an explicit (Euclidean) metric:
  - Linearity and product rules for the gradient
  - Divergence identities involving products, cross products, and advection terms
  - Curl relations, including curl-of-curl and divergence-of-curl checks
  - Laplacian formulations expressed via gradient, divergence, and raised/lowered indices

### 4. *cov_operators_validation.ipynb*
  Extends the validation to covariant calculus on curvilinear coordinates using a `ConnectOps` connection:
  - Covariant derivative, divergence, and curl driven by Christoffel symbols
  - Directional and Lie derivatives of tensor fields
  - Consistency with the metric-aware Cartesian operators as a limiting case


### Running Tutorials

```bash
# Launch Jupyter
jupyter notebook

# Navigate to nb/tutorials/
cd nb/tutorials

# Open any .ipynb file to get started
```

---

## 🧪 Scientific ML Examples

The `nb/examples/` directory contains production-ready implementations of advanced scientific machine learning techniques:

### 1. *Physics-Informed Neural Networks (PINN)*
  Underdamped oscillator model combining sparse data with ODE residuals, using TensOps finite-difference derivatives and autograd for physics-aligned training.

### 2. *Structure-Preserving Neural Networks (SPNN)*
  Double pendulum integrator that preserves geometric degeneracy via TensOps contractions and residual-based penalties.


### 3. *Physics-Guided Neural Network Invariants (PGNNIV)*
  Nonlinear elasticity solver coupling encoder-decoder and invariant layers, leveraging TensOps calculus operators ( gradient and contraction layers, and divergence) for preprocessing, feature extraction, and loss evaluations.


### Running Examples

Each example includes a `model/main.ipynb` notebook that demonstrates the complete workflow:

```bash
# Navigate to an example
cd nb/examples/pgnniv/

# Launch the main notebook
jupyter notebook model/main.ipynb
```

---


## 🔧 Configuration

### Backend Configuration

Switch between different backends:

```python
from tensopsciml.backends import get_backend, set_backend

# Get current backend
backend = get_backend()

# Set specific backend (future extensions)
# set_backend('torch')  # PyTorch (default)
```

### Tensor Configuration

```python
from tensopsciml.tensors.config import TensorConfig

# Describe tensor metadata once and reuse during TensOps creation
config = TensorConfig(
    domain_dim: int
    codomain_dim: list[int]
    order: list[int]
    contravariant_indices: list[int] | None = None
    covariant_indices: list[int] | None = None
)
```

---

## 🧪 Testing

Comprehensive test suite with 200+ tests covering:
- Tensor operations and algebra
- Calculus operators
- Finite difference kernels
- Backend integration
- Utility functions

### Run All Tests

```bash
# Using pytest (recommended)
python -m pytest tests/ -v

# Using unittest
python -m unittest discover -s tests -p "test_*.py" -v
```

### Run Specific Test Suites

```bash
# Test tensor algebra
python -m pytest tests/algebra/ -v

# Test kernels
python -m pytest tests/kernels/ -v

# Test backends
python -m pytest tests/backends/ -v

# Test with coverage
python -m pytest tests/ -v --cov=src/tensopsciml --cov-report=html
```

---

## 📊 Performance Considerations

### GPU Acceleration

All operations automatically support GPU computation via PyTorch:

```python
import torch
from tensopsciml import TensOps, Derivative, partial_derivative

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

values = torch.randn(1, 2, 64, 64, device=device)
vector = TensOps(
  values,
  domain_dim=2,
  codomain_dim=[2],
  order=[1, 0],
  contravariant_indices=[0],
  covariant_indices=None,
)

kernels = Derivative(d1=0.05, d2=0.05, device=device).two_dimensions()
grad_vector = partial_derivative(vector, kernels)
```

---

## 🤝 Contributing

Contributions are welcome! By participating in this project, you agree that your
contributions will be licensed under the same license as the project (Apache License 2.0). 

Please ensure:

1. **Code Quality**: Follow Python PEP 8 and use type hints
2. **Documentation**: Google Style docstrings for all functions and classes
3. **Testing**: All new features must include comprehensive tests
4. **Examples**: Complex features should include usage examples

### Development Setup

```bash
# Clone repository
git clone <repository-url>

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/ -v
```

---

## 📖 Documentation

Full API documentation is available in:
- **Inline docstrings**: Hover over functions and classes in your IDE.
- **Tutorial notebooks**: Hands-on guides in [nb/tutorials](nb/tutorials).
- **Scientific ML examples**: End-to-end case studies in [nb/examples](nb/examples).

---

## 📄 License

This project is released under the Apache License 2.0 (see the [LICENSE](LICENSE) file). It is part of the TensOps computational framework developed at the Tissue Microenvironment Lab (TME Lab), Aragón Institute for Engineering Research (I3A), University of Zaragoza.

---

## ✨ Key Highlights

- **Scientific Computing**: Designed for researchers and practitioners in Scientific Machine Learning
- **Flexible Architecture**: Backend-agnostic design allows integration with different computational frameworks
- **High-Order Tensors**: Emphasizes rigorous tensor calculus with explicit index metadata across arbitrary tensor orders
- **Differential Operators**: Provides convolution-based derivative operators that generalize beyond finite differences while remaining differentiable
- **General Geometry**: Explicit, configurable metric tensors, volume forms, and Levi-Civita connections enable covariant calculus beyond the Cartesian case
- **Production-Ready**: Used in published research on physics-informed neural networks
- **Well-Documented**: Comprehensive docstrings, tutorials, and examples
- **Tested**: Extensive test coverage ensures reliability

---

## 🚀 Getting Started

1. **Install TensOps**: Follow the [Installation](#-installation) section
2. **Run Tutorials**: Start with `tensops_tutorial.ipynb` in `nb/tutorials/`
3. **Explore Examples**: Check out application examples in `nb/examples/`
5. **Run Tests**: Verify installation with `python -m pytest tests/ -v`

---

## 📧 Support

For issues, questions, or contributions, please refer to the project repository or contact Rubén Muñoz, who maintains the codebase, at rmunoz@unizar.es.

---

## 🎓 Citation

If you use TensOps in your research, please cite this project appropriately. See your institution's guidelines for citing software in academic work.

---
