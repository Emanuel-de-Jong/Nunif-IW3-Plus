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


def reglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=dim)
    return F.relu(x1) * x2


class ReGLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return reglu(x, dim=self.dim)


def tanhglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=dim)
    return F.tanh(x1) * x2


class TanhGLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return tanhglu(x, dim=self.dim)


def singlu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=dim)
    return torch.sin(x1) * x2


class SinGLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return singlu(x, dim=self.dim)


def spanglu(x: torch.Tensor | tuple[torch.Tensor, torch.Tensor], dim: int = -1) -> torch.Tensor:
    if torch.is_tensor(x):
        x1, x2 = x.chunk(2, dim=dim)
    else:
        x1, x2 = x
    return (torch.sigmoid(x1) - 0.5) * x2


class SPANGLU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return spanglu(x, dim=self.dim)


def _test():
    dim = 32
    dim2 = dim * 2

    compute_swiglu_dim(32) == 48
    compute_swiglu_dim(32, factor=0.1) == 8

    model = nn.Sequential(
        nn.Linear(dim, dim2),
        SwiGLU(),
        nn.Linear(dim, dim2),
        ReGLU(),
        nn.Linear(dim, dim2),
        TanhGLU(),
        nn.Linear(dim, dim2),
        SinGLU(),
    ).cuda()
    x = torch.zeros((4, 32)).cuda()
    model(x).sum().backward()

    span = SPANGLU(dim=1).cuda()
    x = torch.rand((4, 32, 8, 8)).cuda()
    shortcut = torch.rand_like(x)
    z1 = span(torch.cat([x, x + shortcut], dim=1))
    z2 = span((x, x + shortcut))
    assert torch.all(torch.isclose(z1, z2))


if __name__ == "__main__":
    _test()
