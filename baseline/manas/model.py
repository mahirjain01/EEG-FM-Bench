from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn
import torch


@dataclass(frozen=True)
class MAEConfig:
    # Data
    fs: int = 200
    patch_size: int = 200
    overlap_size: int = 20

    # Model:
    dim: int = 512
    enc_layers: int = 12
    enc_heads: int = 8
    dec_layers: int = 4
    dec_heads: int = 8

    # Masking
    mask_ratio: float = 0.55
    spat_radius: float = 3.0
    time_radius: float = 3.0

    # Loss
    aux_weight: float = 0.1

    # Coord Encoding
    n_freqs: int = 4

    @property
    def stride(self):
        return self.patch_size - self.overlap_size


@dataclass
class MAEOut:
    loss: Tensor
    loss_key: Tensor
    loss_aux: Tensor


# ==================================
# Symbols
# ==================================

# B: Batch
# C: Channels
# T: Time
# P: Patches Per Channel
# N: Total Tokens (C * P)
# D: Embedding Dim
# S: Patch Size

# ==================================
# Utils
# ==================================


class GEGLU(nn.Module):
    def __init__(self, dim: int = 512, mult: int = 4):
        super().__init__()

        hidden = dim * mult

        self.fc = nn.Linear(dim, hidden * 2)
        self.project = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        x, gate = self.fc(x).chunk(2, dim=-1)
        return self.project(x * torch.nn.functional.gelu(gate))


# ==================================
# Tokenizer
# ==================================


class PatchEmbed(nn.Module):
    """
    Patch Tokenizer with Linear Projection.

    Input
        x: (B, C, T)

    Output
        tok: (B, C, P, D)
        pat: (B, C, P, S)
    """

    def __init__(self, patch_size: int = 200, stride: int = 180, dim: int = 512):
        super().__init__()

        self.patch_size = patch_size
        self.stride = stride

        self.project = nn.Linear(self.patch_size, dim, bias=False)

    def patchify(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected input of shape (B, C, T), got {x.shape}.")
        return x.unfold(-1, self.patch_size, self.stride)  # (B, C, P, S)

    def forward(self, x: Tensor) -> Tensor:
        pat = self.patchify(x)  # (B, C, P, S)
        tok = self.project(pat)  # (B, C, P, D)
        return tok, pat


# ==================================
# Positional Encoding
# ==================================


class CoordEnc(nn.Module):
    """
    4D Coordinate Encoder for (x, y, z, t).

    Input
        coords: (B, N, 4)

    Output
        pos_enc: (B, N, D)
    """

    def __init__(self, n_freqs: int = 4, dim: int = 512):
        super().__init__()

        freqs = torch.linspace(1.0, 10.0, n_freqs)
        basis = torch.cartesian_prod(freqs, freqs, freqs, freqs).transpose(1, 0)

        self.register_buffer("basis", basis)

        self.fourier_project = nn.Linear(2 * (n_freqs**4), dim, bias=False)
        self.coord_project = nn.Sequential(
            nn.Linear(4, dim, bias=False),
            nn.GELU(),
            nn.RMSNorm(dim, dtype=torch.bfloat16),
        )
        self.norm = nn.RMSNorm(dim, dtype=torch.bfloat16)

    def forward(self, coords: Tensor) -> Tensor:
        if coords.ndim != 3 or coords.shape[-1] != 4:
            raise ValueError(f"expected input of shape (B, N, 4), got {coords.shape}.")

        phase = coords @ self.basis
        fourier = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        pos_enc = self.fourier_project(fourier) + self.coord_project(coords)
        return self.norm(pos_enc)


class TransformerBlock(nn.Module):
    """
    Standard Pre-Norm Transformer Block.

    Input / Output
        x: (B, N, D)
        ffn_out: (B, N, D)
    """

    def __init__(self, heads: int, dim: int = 512):
        super().__init__()

        if dim % heads != 0:
            raise ValueError(
                f"embedding dimension {dim} must be divisible by number of heads {heads}."
            )

        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5

        self.attn_norm = nn.RMSNorm(dim, dtype=torch.bfloat16)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_project = nn.Linear(dim, dim, bias=False)

        self.ffn_norm = nn.RMSNorm(dim, dtype=torch.bfloat16)
        self.ffn = GEGLU(dim, mult=4)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, N, D = x.shape

        x_norm = self.attn_norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.heads, self.head_dim).transpose(1, 2)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)

        x = x + self.out_project(attn_out)

        ffn_in = self.ffn_norm(x)
        ffn_out = self.ffn(ffn_in)

        x = x + ffn_out
        return x, ffn_out


class TransformerStack(nn.Module):
    """
    Transformer Stack.

    Input
        x: (B, N, D)

    Output
        x: (B, N, D)
        ffns: List[(B, N, D)]
    """

    def __init__(
        self,
        depth: int,
        heads: int,
        dim: int = 512,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [TransformerBlock(heads, dim) for _ in range(depth)]
        )
        self.norm = nn.RMSNorm(dim, dtype=torch.bfloat16)

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        ffns: list[Tensor] = []

        for blk in self.blocks:
            x, ffn_out = blk(x)
            ffns.append(ffn_out)

        return self.norm(x), ffns


# ==================================
# Decoder
# ==================================


class Decoder(nn.Module):
    """
    MAE Decoder.

    Input
        vis: (B, N_vis, D)
        coords: (B, N, 4)
        mask: (B, N), True = visible

    Output
        recon: (B, N, S)
    """

    def __init__(self, depth: int, heads: int, dim: int = 512, patch_size: int = 200):
        super().__init__()

        self.mask_tok = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_tok, std=0.02)

        self.stack = TransformerStack(depth, heads, dim)
        self.predict = nn.Linear(dim, patch_size)

    def forward(self, vis: Tensor, coords_enc: Tensor, mask: Tensor) -> Tensor:
        B, N = mask.shape
        D = vis.shape[-1]

        full = (
            self.mask_tok.to(device=vis.device, dtype=vis.dtype).expand(B, N, D).clone()
        )
        full[mask] = vis.reshape(-1, D)

        dec, _ = self.stack(full + coords_enc)

        return self.predict(dec)


# ==================================
# Main MAE Model
# ==================================


class MAE(nn.Module):
    """
    Masked Autoencoder for EEG-like channel x time data.

    Input
        x: (B, C, T)
        xyz: (B, N, 3)
    """

    def __init__(self, cfg: MAEConfig):
        super().__init__()

        self.cfg = cfg

        self.patch = PatchEmbed(cfg.patch_size, cfg.stride, cfg.dim)
        self.coord_enc = CoordEnc(cfg.n_freqs, cfg.dim)

        self.enc = TransformerStack(cfg.enc_layers, cfg.enc_heads, cfg.dim)
        self.dec = Decoder(cfg.dec_layers, cfg.dec_heads, cfg.dim, cfg.patch_size)

        aux_dim = cfg.dim * cfg.enc_layers
        self.aux_query = nn.Parameter(torch.randn(1, 1, aux_dim))
        nn.init.normal_(self.aux_query, std=0.02)

        self.aux_project = nn.Linear(aux_dim, cfg.dim)
        self.aux_predict = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim),
            nn.GELU(),
            nn.Linear(cfg.dim, cfg.patch_size),
        )

    def _build_coords(self, xyz: Tensor, P: Tensor) -> Tensor:
        """
        Input
            xyz: (B, C, 3)
            P: patches per channel, scalar
        Output
            coords: (B, N, 4)
        """

        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"expected input of shape (B, N, 3), got {xyz.shape}.")

        B, C, _ = xyz.shape
        device = xyz.device

        t = torch.arange(P, device=device, dtype=torch.float32)  # (P,)
        t = t.view(1, 1, P, 1).expand(B, C, P, 1)  # (B, C, P, 1)

        xyz = xyz.unsqueeze(2).expand(-1, -1, P, -1)  # (B, C, P, 3)
        coords = torch.cat([xyz, t], dim=-1)  # (B, C, P, 4)

        return coords.flatten(1, 2)  # (B, N, 4)

    def _flatten_tokens(self, tok: Tensor, pat: Tensor) -> tuple[Tensor, Tensor]:
        """
        Input:
            tok: (B, C, P, D)
            pat: (B, C, P, S)
        Output:
            tok_flat: (B, N, D)
            pat_flat: (B, N, S)
        """
        tok = tok.flatten(1, 2)  # (B, N, D)
        pat = pat.flatten(1, 2)  # (B, N, S
        return tok, pat

    def forward(self, x: Tensor, xyz: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected input of shape (B, C, T), got {x.shape}.")

        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"expected input of shape (B, N, 3), got {xyz.shape}.")

        if x.shape[:2] != xyz.shape[:2]:
            raise ValueError(
                f"batch and channel dimensions of x and xyz must match, got {x.shape} and {xyz.shape}."
            )

        tok, pat = self.patch(x)  # (B, C, P, D), (B, C, P, S)
        P = tok.shape[2]

        tok, _ = self._flatten_tokens(tok, pat)  # (B, N, D), (B, N, S)

        coords = self._build_coords(xyz, P)  # (B, N, 4)
        coords_enc = self.coord_enc(coords)

        tok = tok + coords_enc

        z, ffns = self.enc(tok)  # (B, N_vis, D), List[(B, N_vis, D)]

        return z
