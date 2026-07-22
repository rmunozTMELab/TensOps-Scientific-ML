"""Dataclass bundling the metadata that describes a TensOps tensor."""

from dataclasses import dataclass


@dataclass(slots=True)
class TensorConfig:
    """Configuration dataclass for TensOps tensor metadata.

    Attributes:
        domain_dim (int): Spatial discretization dimensionality.
        codomain_dim (list[int]): Tensor codomain dimensions describing
            rank-specific axes.
        order (list[int]): Pair ``[contravariant, covariant]`` describing
            tensorial order.
        contravariant_indices (list[int] | None): Positions of contravariant
            (upper) indices.
        covariant_indices (list[int] | None): Positions of covariant (lower)
            indices.
    """

    domain_dim: int
    codomain_dim: list[int]
    order: list[int]
    contravariant_indices: list[int] | None = None
    covariant_indices: list[int] | None = None
