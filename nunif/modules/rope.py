import torch
import torch.nn as nn


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    i1 = (Ellipsis, slice(None, half))
    i2 = (Ellipsis, slice(half, None))
    if True:
        # This requires Torch 2.13 or later.
        sin = sin[i1]  # sin[i1] == sin[i2]
        # float32
        out = x * cos
        out[i1].addcmul_(x[i2], sin, value=-1)
        out[i2].addcmul_(x[i1], sin, value=1)
        return out
    else:
        out = torch.empty_like(x)
        sin = sin[i1]  # sin[i1] == sin[i2]
        cos = cos[i1]
        out[i1] = x[i1] * cos - x[i2] * sin
        out[i2] = x[i2] * cos + x[i1] * sin
        return out


class RoPE2d(nn.Module):
    cos_h: torch.Tensor
    sin_h: torch.Tensor
    cos_w: torch.Tensor
    sin_w: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        height: int,
        width: int,
        base: float = 100.0,
    ) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be a multiple of 4"

        self.head_dim = head_dim
        self.height = height
        self.width = width
        self.dim_chunk = head_dim // 2

        cos_h, sin_h, cos_w, sin_w = self._precompute_freqs(base)
        self.register_buffer("cos_h", cos_h, persistent=False)
        self.register_buffer("sin_h", sin_h, persistent=False)
        self.register_buffer("cos_w", cos_w, persistent=False)
        self.register_buffer("sin_w", sin_w, persistent=False)

    @torch.no_grad()
    def _precompute_freqs(self, base: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cos_h, sin_h = self._get_1d_sin_cos(base, self.height, self.dim_chunk)
        cos_w, sin_w = self._get_1d_sin_cos(base, self.width, self.dim_chunk)

        cos_h = cos_h.reshape(1, 1, self.height, 1, self.dim_chunk)
        sin_h = sin_h.reshape(1, 1, self.height, 1, self.dim_chunk)
        cos_w = cos_w.reshape(1, 1, 1, self.width, self.dim_chunk)
        sin_w = sin_w.reshape(1, 1, 1, self.width, self.dim_chunk)

        return cos_h, sin_h, cos_w, sin_w

    def _get_1d_sin_cos(self, base: float, max_len: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        exp = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        inv_freq = 1.0 / (base**exp)
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_heads, N, head_dim] (N = h * w)
        """
        B, num_heads, N, head_dim = x.shape
        x_h = x[..., : self.dim_chunk].reshape(B, num_heads, self.height, self.width, self.dim_chunk)
        x_w = x[..., self.dim_chunk :].reshape(B, num_heads, self.height, self.width, self.dim_chunk)

        out_h = apply_rope(x_h, self.cos_h.to(x.dtype), self.sin_h.to(x.dtype))
        out_w = apply_rope(x_w, self.cos_w.to(x.dtype), self.sin_w.to(x.dtype))

        out_h = out_h.reshape(B, num_heads, N, self.dim_chunk)
        out_w = out_w.reshape(B, num_heads, N, self.dim_chunk)
        return torch.cat([out_h, out_w], dim=-1)


def _bench(do_compile):
    import time

    device = "cuda:0"
    N = 20
    LAYERS = 10
    B = 16
    S = (128, 128)
    dim = 256
    num_heads = 4
    head_dim = dim // 4

    rope = RoPE2d(head_dim, S[0], S[1]).cuda()
    x = torch.rand((B, num_heads, S[0] * S[1], head_dim), dtype=torch.float16, device="cuda")
    if do_compile:
        # over 10x faster
        rope = torch.compile(rope)

    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        z = rope(x)
        print(z.shape, z.dtype)

    # check backward works
    rope(x + nn.Parameter(torch.zeros(1)).cuda()).sum().backward()

    # benchmark
    torch.cuda.synchronize()
    t = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for _ in range(N):
            for _ in range(LAYERS):
                z = rope(x)
    torch.cuda.synchronize()
    print(round(1 / ((time.time() - t) / (B * N)), 3), "FPS")
    max_vram_mb = int(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
    print(f"GPU Max Memory Allocated {max_vram_mb}MB")


if __name__ == "__main__":
    _bench(do_compile=False)
    _bench(do_compile=True)
