from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class MAEConfig:
    fs: int = 200
    patch_size: int = 200
    overlap_size: int = 20
    dim: int = 512
    enc_layers: int = 12
    enc_heads: int = 8
    dec_layers: int = 4
    dec_heads: int = 8
    mask_ratio: float = 0.55
    aux_weight: float = 0.1
    n_freqs: int = 4
    time_scale_factor: float = 1.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        norm = x.norm(2, dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        return (x / (norm + self.eps)) * self.weight


class GEGLUFFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hidden_dim = dim * 4
        self.in_proj = nn.Linear(dim, 2 * hidden_dim, bias=True)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x: Tensor):
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(x * F.gelu(gate))


class GEGLUProject(nn.Module):
    def forward(self, x: Tensor):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FlashAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: Tensor):
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv_proj(x).reshape(
            batch_size, num_tokens, 3, self.heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.pre_attn_norm = RMSNorm(dim)
        self.attn = FlashAttention(dim, heads)
        self.pre_ffn_norm = RMSNorm(dim)
        self.ffn = GEGLUFFN(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.pre_attn_norm(x))
        x = x + self.ffn(self.pre_ffn_norm(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        fs: int = 200,
        patch_size: int = 200,
        overlap_size: int = 20,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.step = self.patch_size - overlap_size
        self.linear = nn.Linear(self.patch_size, embed_dim, bias=False)


class PosEnc4D(nn.Module):
    def __init__(self, n_freqs: int = 4, embed_dim: int = 512, noise_std: float = 0.25):
        super().__init__()
        self.noise_std = noise_std

        freqs = torch.linspace(1.0, 10.0, n_freqs)
        self.register_buffer(
            "freq_matrix",
            torch.cartesian_prod(freqs, freqs, freqs, freqs).transpose(1, 0),
        )

        self.fourier_linear = nn.Linear(2 * (n_freqs**4), embed_dim, bias=False)
        self.learned_linear = nn.Sequential(
            nn.Linear(4, embed_dim * 2, bias=False),
            GEGLUProject(),
            RMSNorm(embed_dim),
        )
        self.final_norm = RMSNorm(embed_dim)

    def forward(self, coords: Tensor):
        if self.training:
            noise = torch.randn_like(coords[:, :, :3]) * self.noise_std
            coords = coords.clone()
            coords[:, :, :3] += noise

        phases = torch.matmul(coords, self.freq_matrix)
        fourier_features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        fourier_features = self.fourier_linear(fourier_features)
        learned_features = self.learned_linear(coords)
        return self.final_norm(fourier_features + learned_features)


class Encoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(depth)]
        )
        self.final_norm = RMSNorm(dim)

    def forward(self, x: Tensor):
        intermediates = []
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)
        return self.final_norm(x), intermediates


class DecoderCore(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, heads) for _ in range(depth)]
        )
        self.final_norm = RMSNorm(dim)

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class Decoder(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, patch_size: int):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder = DecoderCore(dim, depth, heads)
        self.predict = nn.Linear(dim, patch_size, bias=True)


class MAE(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbed(
            fs=cfg.fs,
            patch_size=cfg.patch_size,
            overlap_size=cfg.overlap_size,
            embed_dim=cfg.dim,
        )
        self.pos_enc = PosEnc4D(n_freqs=cfg.n_freqs, embed_dim=cfg.dim)
        self.encoder = Encoder(cfg.dim, cfg.enc_layers, cfg.enc_heads)
        self.decoder = Decoder(cfg.dim, cfg.dec_layers, cfg.enc_heads, cfg.patch_size)

        self.aux_dim = cfg.enc_layers * cfg.dim
        self.aux_query = nn.Parameter(torch.randn(1, 1, self.aux_dim))
        nn.init.normal_(self.aux_query, std=0.02)

        self.aux_linear = nn.Linear(self.aux_dim, cfg.dim, bias=False)
        self.aux_predict = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim * 2, bias=True),
            GEGLUProject(),
            nn.Linear(cfg.dim, cfg.patch_size, bias=True),
        )


class MAEEncoderAdapter(nn.Module):
    """
    Benchmark-facing wrapper that preserves the checkpoint's `mae.*` key layout.
    """

    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.mae = MAE(cfg)

    def forward(self, x: Tensor, xyz: Tensor) -> Tensor:
        batch_size, num_channels, _ = x.shape
        device = x.device

        patches = x.unfold(
            -1,
            self.mae.patch_embed.patch_size,
            self.mae.patch_embed.step,
        )
        num_patches = patches.shape[2]
        tokens = self.mae.patch_embed.linear(patches).flatten(1, 2)

        time_idx = (
            torch.arange(num_patches, device=device, dtype=torch.float32)
            * self.mae.cfg.time_scale_factor
        )
        spatial = xyz.unsqueeze(2).expand(-1, -1, num_patches, -1)
        temporal = time_idx.view(1, 1, num_patches, 1).expand(
            batch_size, num_channels, -1, -1
        )
        coords = torch.cat([spatial, temporal], dim=-1).flatten(1, 2)

        encoded = tokens + self.mae.pos_enc(coords)
        encoded, _ = self.mae.encoder(encoded)
        return encoded
