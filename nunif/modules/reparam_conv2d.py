import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import basic_module_init
from .replication_pad2d import ReplicationPad2dNaive


def _fit_to_size(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if x.shape[-2] == h and x.shape[-1] == w:
        return x
    dy = (x.shape[-2] - h) // 2
    dx = (x.shape[-1] - w) // 2
    return x[..., dy : dy + h, dx : dx + w]


class ReparamBranch(nn.Module):
    def fuse(self):
        raise NotImplementedError

    def get_kernel_size(self):
        raise NotImplementedError


class Conv2dBranch(ReparamBranch):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

    def forward(self, x):
        return self.conv(x)

    def fuse(self):
        w = self.conv.weight.double()
        b = (
            self.conv.bias.double()
            if self.conv.bias is not None
            else torch.zeros(self.out_channels, dtype=torch.double, device=w.device)
        )
        return w, b

    def get_kernel_size(self):
        return self.kernel_size


class Parallel(ReparamBranch):
    def __init__(self, in_channels: int, out_channels: int, structures):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.branches = nn.ModuleList([self._build(in_channels, out_channels, s) for s in structures])
        self.gate = nn.Parameter(torch.zeros(len(structures)))

    def _build(self, in_c, out_c, s):
        if isinstance(s, (int, tuple)):
            return Conv2dBranch(in_c, out_c, s)
        if isinstance(s, ReparamBranch):
            return s
        raise ValueError(f"Unsupported structure: {s}")

    def forward(self, x):
        outputs = [b(x) for b in self.branches]
        min_h = min(o.shape[-2] for o in outputs)
        min_w = min(o.shape[-1] for o in outputs)

        res = 0
        gate = self.gate.to(outputs[0].dtype)
        for i, o in enumerate(outputs):
            o = _fit_to_size(o, min_h, min_w)
            res = res + (1.0 + gate[i]) * o
        return res

    def fuse(self):
        fused = [b.fuse() for b in self.branches]
        max_h, max_w = self.get_kernel_size()

        res_w = torch.zeros(
            (self.out_channels, self.in_channels, max_h, max_w), dtype=torch.double, device=self.gate.device
        )
        res_b = torch.zeros(self.out_channels, dtype=torch.double, device=self.gate.device)

        gate = self.gate.double()
        for i, (w, b) in enumerate(fused):
            ph = (max_h - w.shape[-2]) // 2
            pw = (max_w - w.shape[-1]) // 2
            res_w += (1.0 + gate[i]) * F.pad(w, [pw, pw, ph, ph])
            res_b += (1.0 + gate[i]) * b
        return res_w, res_b

    def get_kernel_size(self):
        sizes = [b.get_kernel_size() for b in self.branches]
        return max(s[0] for s in sizes), max(s[1] for s in sizes)


class Series(ReparamBranch):
    def __init__(self, in_channels: int, out_channels: int, structures, middle_factor: int | float = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Scaled relative to out_channels
        self.middle_channels = int(out_channels * middle_factor)
        self.middle_channels = max(self.middle_channels - self.middle_channels % 8, 8)

        branches = []
        num_branches = len(structures)
        for i, s in enumerate(structures):
            branch_in = in_channels if i == 0 else self.middle_channels
            branch_out = out_channels if i == num_branches - 1 else self.middle_channels
            branches.append(self._build(branch_in, branch_out, s))
        self.branches = nn.ModuleList(branches)

    def _build(self, in_c, out_c, s):
        if isinstance(s, (int, tuple)):
            return Conv2dBranch(in_c, out_c, s)
        if isinstance(s, ReparamBranch):
            return s
        raise ValueError(f"Unsupported structure: {s}")

    def forward(self, x):
        for b in self.branches:
            x = b(x)
        return x

    def fuse(self):
        curr_w, curr_b = self.branches[0].fuse()
        for i in range(1, len(self.branches)):
            next_w, next_b = self.branches[i].fuse()
            curr_w = F.conv2d(
                curr_w.transpose(0, 1), next_w.flip(-1, -2), padding=(next_w.shape[-2] - 1, next_w.shape[-1] - 1)
            ).transpose(0, 1)
            curr_b = next_w.sum(dim=(2, 3)) @ curr_b + next_b
        return curr_w, curr_b

    def get_kernel_size(self):
        sizes = [b.get_kernel_size() for b in self.branches]
        h = sum(s[0] for s in sizes) - (len(sizes) - 1)
        w = sum(s[1] for s in sizes) - (len(sizes) - 1)
        return h, w


class ReparamConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, structure, padding: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(structure, (list, tuple)):
            self.structure = Parallel(in_channels, out_channels, structure)
        else:
            self.structure = structure

        kernel_size = self.structure.get_kernel_size()
        self.padding = (
            ReplicationPad2dNaive((kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] // 2, kernel_size[0] // 2))
            if padding
            else None
        )

        self.conv_eval = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.conv_eval.requires_grad_(False)

        basic_module_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding is not None:
            x = self.padding(x)

        if not self.training:
            return self.conv_eval(x)

        return self.structure(x)

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            return self.fuse()
        return self

    def fuse(self) -> nn.Conv2d:
        assert self.conv_eval.bias is not None
        w, b = self.structure.fuse()
        dtype = self.conv_eval.weight.dtype
        self.conv_eval.weight.data.copy_(w.to(dtype))
        self.conv_eval.bias.data.copy_(b.to(dtype))
        return self.conv_eval

    def purge(self) -> nn.Module:
        delattr(self, "structure")
        return self


def apply_fuse_(model: nn.Module):
    for name, child in model.named_children():
        if isinstance(child, ReparamConv2d):
            setattr(model, name, child.fuse())
        else:
            apply_fuse_(child)


def _test(dtype=torch.float64):
    print("******", dtype)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_c, out_c = 3, 64

    # Structure definition with middle_factor relative to out_channels
    structure = [1, 3, 5, 7, (1, 7), (7, 1), Series(in_c, out_c, [1, 3, 1], middle_factor=2.0)]

    model = ReparamConv2d(in_c, out_c, structure, padding=True).to(device, dtype=dtype)
    print("*** train", model)
    x = torch.rand((4, in_c, 64, 64)).to(device, dtype=dtype)

    if isinstance(model.structure, Parallel):
        with torch.no_grad():
            model.structure.gate.copy_(torch.randn(len(model.structure.branches)))

    model.train()
    z1 = model(x)

    model.eval()
    model.purge()
    print("*** eval", model)
    z2 = model(x)

    diff = (z1 - z2).abs().max().item()
    print(f"Max difference: {diff}")
    if diff < 1e-5:
        print("Test Passed")
    else:
        print("Test Failed")


if __name__ == "__main__":
    _test(torch.float64)
    # _test(torch.float32)
    # _test(torch.float16)
    # _test(torch.bfloat16)
