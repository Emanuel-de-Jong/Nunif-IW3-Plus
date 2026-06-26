import torch
import torch.nn as nn
import torch.nn.functional as F


def swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    assert x.shape[dim] % 2 == 0
    x1, x2 = x.chunk(2, dim=dim)
    # DINOv2 order; differs from F.glu
    return F.silu(x1) * x2


class SwiGLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x, dim=self.dim)


def compute_swiglu_dim(dim: int, factor: float = 1.5, align_to: int = 8) -> int:
    swiglu_dim = int(dim * factor)
    swiglu_dim -= swiglu_dim % align_to
    swiglu_dim = max(swiglu_dim, align_to)
    return swiglu_dim


def _test():
    dim = 32
    swiglu_dim = compute_swiglu_dim(dim)
    model = nn.Sequential(
        nn.Linear(dim, swiglu_dim * 2),
        SwiGLU(),
        nn.Linear(swiglu_dim, dim),
    ).cuda()
    x = torch.zeros((4, 32)).cuda()
    model(x).sum().backward()

    compute_swiglu_dim(32) == 48
    compute_swiglu_dim(32, factor=0.1) == 8


if __name__ == "__main__":
    _test()
