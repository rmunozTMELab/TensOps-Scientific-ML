import torch

from tensopsciml import TensOps


def mae(a, b):
    """Compute the mean absolute error (MAE) between two inputs.

    This function accepts either `TensOps` objects or plain PyTorch tensors.
    If a `TensOps` instance is provided, the underlying tensor stored in its
    `.values` attribute will be used. The function computes the element-wise
    absolute difference and returns the mean over all elements as a scalar
    `torch.Tensor`.

    Args:
        a (Union[TensOps, torch.Tensor, object]): First input. If a
            `TensOps` instance is passed the `.values` attribute is used. If
            another object is passed it is expected to provide a `.values`
            attribute referencing a `torch.Tensor`.
        b (Union[TensOps, torch.Tensor, object]): Second input. Behavior
            mirrors that of ``a``.

    Returns:
        torch.Tensor: Scalar tensor containing the mean absolute error.

    Raises:
        AttributeError: If neither input provides a tensor (neither a
            `torch.Tensor` nor exposes a `.values` attribute).

    Examples:
        >>> mae(torch.tensor([1., 2.]), torch.tensor([1.5, 1.]))
        tensor(0.7500)

        >>> # With TensOps wrapper:
        >>> t1 = TensOps(torch.tensor([[[1., 2.]]]), domain_dim=1, codomain_dim=1, contravariance=0, covariance=0)
        >>> t2 = TensOps(torch.tensor([[[1.5, 1.]]]), domain_dim=1, codomain_dim=1, contravariance=0, covariance=0)
        >>> mae(t1, t2)
        tensor(0.7500)
    """

    # Helper to extract a torch.Tensor from TensOps or objects exposing `.values`.
    def _extract(x):
        if isinstance(x, TensOps):
            return x.values
        if isinstance(x, torch.Tensor):
            return x
        maybe = getattr(x, "values", None)
        if isinstance(maybe, torch.Tensor):
            return maybe
        raise AttributeError("Input must be a torch.Tensor, TensOps, or expose a .values tensor")

    a_ = _extract(a)
    b_ = _extract(b)

    E = torch.abs(a_ - b_)
    
    return torch.mean(E)