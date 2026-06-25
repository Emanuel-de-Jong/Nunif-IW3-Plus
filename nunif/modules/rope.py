import torch
import torch.nn as nn


class RoPE2d(nn.Module):
    cos_h: torch.Tensor
    sin_h: torch.Tensor
    cos_w: torch.Tensor
    sin_w: torch.Tensor

    def __init__(self, head_dim: int, height: int, width: int, theta: float = 100.0) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be a multiple of 4"
        self.head_dim = head_dim
        self.theta = theta
        self.height = height
        self.width = width

        cos_h, sin_h, cos_w, sin_w = self._precompute_freqs(head_dim, height, width)
        self.register_buffer("cos_h", cos_h, persistent=False)
        self.register_buffer("sin_h", sin_h, persistent=False)
        self.register_buffer("cos_w", cos_w, persistent=False)
        self.register_buffer("sin_w", sin_w, persistent=False)

    @torch.no_grad()
    def _precompute_freqs(self, head_dim: int, h: int, w: int):
        dim_h = head_dim // 2
        dim_w = head_dim // 2

        cos_h_1d, sin_h_1d = self._get_1d_sin_cos(h, dim_h)
        cos_w_1d, sin_w_1d = self._get_1d_sin_cos(w, dim_w)

        cos_h = cos_h_1d[:, None].repeat(1, w, 1).reshape(1, 1, h * w, dim_h)
        sin_h = sin_h_1d[:, None].repeat(1, w, 1).reshape(1, 1, h * w, dim_h)

        cos_w = cos_w_1d[None, :].repeat(h, 1, 1).reshape(1, 1, h * w, dim_w)
        sin_w = sin_w_1d[None, :].repeat(h, 1, 1).reshape(1, 1, h * w, dim_w)

        return cos_h.contiguous(), sin_h.contiguous(), cos_w.contiguous(), sin_w.contiguous()

    def _get_1d_sin_cos(self, max_len: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        exp = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        inv_freq = 1.0 / (self.theta**exp)
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_heads, N, head_dim] (N = h * w)
        """
        assert x.shape[-1] == self.head_dim
        x_float = x.to(torch.float32)
        x_h = x_float[..., : self.head_dim // 2]
        x_w = x_float[..., self.head_dim // 2 :]

        out_h = (x_h * self.cos_h) + (self.rotate_half(x_h) * self.sin_h)
        out_w = (x_w * self.cos_w) + (self.rotate_half(x_w) * self.sin_w)

        return torch.cat([out_h, out_w], dim=-1).to(x.dtype)


def _test():
    B = 4
    num_heads = 4
    h = 8
    w = 4
    head_dim = 32

    rope = RoPE2d(head_dim, h, w).cuda()
    assert rope.cos_w.dtype == torch.float32
    x = torch.rand((B, num_heads, h * w, head_dim)).cuda()
    with torch.autocast(device_type=x.device.type):
        z = rope(x)
    assert x.shape == z.shape

    z = rope(x.to(torch.float16))
    assert z.dtype == torch.float16


if __name__ == "__main__":
    _test()
