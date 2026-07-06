import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import basic_module_init
from .replication_pad2d import ReplicationPad2dNaive


class ParallelConv2d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_sizes: tuple[int | tuple[int, int]], padding: bool = True
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_kernel_size = get_max_kernel_size(kernel_sizes)
        self.scale = len(kernel_sizes) ** -0.5

        self.padding = None
        if padding:
            self.padding = ReplicationPad2dNaive(
                (
                    self.max_kernel_size[1] // 2,
                    self.max_kernel_size[1] // 2,
                    self.max_kernel_size[0] // 2,
                    self.max_kernel_size[0] // 2,
                ),
                detach=True,
            )

        self.gate = nn.Parameter(
            torch.zeros(
                len(kernel_sizes),
            )
        )
        self.conv_modules = nn.ModuleList()
        for ks in kernel_sizes:
            self.conv_modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=ks, padding=0))

        self.conv_eval = nn.Conv2d(in_channels, out_channels, kernel_size=self.max_kernel_size, padding=0)
        self.conv_eval.requires_grad_(False)

        basic_module_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding is not None:
            x = self.padding(x)

        if not self.training:
            return self.conv_eval(x)

        input_size = x.shape[-2:]
        out = sum(
            self.scale * (1.0 + self.gate[i]) * fit_size(conv(x), input_size, self.max_kernel_size)
            for i, conv in enumerate(self.conv_modules)
        )
        return out

    def train(self, mode: bool = True) -> nn.Module:
        super().train(mode)
        if not mode:
            return self.fuse()
        return self

    def purge(self):
        delattr(self, "gate")
        delattr(self, "conv_modules")

    def fuse(self) -> nn.Module:
        dtype = self.gate.dtype
        branches = self.conv_modules
        multipliers = self.scale * (1.0 + self.gate)

        fused_weight, fused_bias = fuse_conv_branches(branches, multipliers, self.max_kernel_size)

        self.conv_eval.weight.data.copy_(fused_weight.to(dtype))
        if self.conv_eval.bias is not None:
            self.conv_eval.bias.data.copy_(fused_bias.to(dtype))

        return self


def fit_size(input_tensor: torch.Tensor, input_size, max_kernel_size) -> torch.Tensor:
    target_h = input_size[0] - (max_kernel_size[0] // 2 * 2)
    target_w = input_size[1] - (max_kernel_size[1] // 2 * 2)

    if input_tensor[-2:] == (target_h, target_w):
        return input_tensor

    dy = input_tensor.shape[-2] - target_h
    dx = input_tensor.shape[-1] - target_w
    assert dy >= 0 and dx >= 0
    assert dy % 2 == 0 and dx % 2 == 0

    y = dy // 2
    x = dx // 2
    return input_tensor[..., y : input_tensor.shape[-2] - y, x : input_tensor.shape[-1] - x]


def fuse_conv_branches(branches, multipliers, target_kernel_size):
    """
    Fuse multiple Conv2d modules into a single weight and bias.
    multipliers: coefficients for each branch (e.g. 1.0 + gate[i])
    target_kernel_size: (H, W) of the resulting kernel
    """
    device = multipliers.device
    out_channels = branches[0].out_channels
    in_channels = branches[0].in_channels

    fused_weight = torch.zeros((out_channels, in_channels, *target_kernel_size), device=device, dtype=torch.double)
    fused_bias = torch.zeros((out_channels,), device=device, dtype=torch.double)

    for conv, coeff in zip(branches, multipliers):
        w = conv.weight.double()
        b = conv.bias.double() if conv.bias is not None else torch.zeros_like(fused_bias)

        kh, kw = w.shape[-2:]
        ph = (target_kernel_size[0] - kh) // 2
        pw = (target_kernel_size[1] - kw) // 2

        fused_weight += coeff * F.pad(w, [pw, pw, ph, ph])
        fused_bias += coeff * b

    return fused_weight, fused_bias


def get_max_kernel_size(kernel_sizes):
    max_h = max_w = 0
    for ks in kernel_sizes:
        if isinstance(ks, int):
            max_h = max(max_h, ks)
            max_w = max(max_w, ks)
        else:
            max_h = max(max_h, ks[0])
            max_w = max(max_w, ks[1])

    return (max_h, max_w)


def _test(dtype):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_channels = 32
    out_channels = 64

    x = torch.rand((4, in_channels, 64, 64)).to(device=device, dtype=dtype)
    for test_case in ((1, 3, 5), (1, (3, 3), (5, 5)), ((1, 1), (1, 3), (1, 7))):
        model = ParallelConv2d(in_channels, out_channels, kernel_sizes=test_case, padding=True).to(
            device=device, dtype=dtype
        )
        print("***", test_case)

        with torch.no_grad():
            model.gate.copy_(torch.randn_like(model.gate))

        model.train()
        z1 = model(x)

        model.eval()
        model.purge()
        z2 = model(x)

        diff = (z1 - z2).abs().max().item()
        mean = (z1 - z2).abs().mean().item()

        print(f"Max difference: {diff}, Mean difference: {mean}")
        if diff < 1e-12:
            print("Test Passed")
        else:
            print("Test Failed")


if __name__ == "__main__":
    _test(dtype=torch.float64)
