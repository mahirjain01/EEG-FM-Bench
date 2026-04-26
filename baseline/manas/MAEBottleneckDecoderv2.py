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

    # Memory
    mem_tokens: int = 8
    mem_layers: int = 2
    query_layers: int = 2

    # Masking
    mask_ratio: float = 0.55
    spat_radius: float = 3.0
    time_radius: float = 3.0

    # Loss
    aux_weight: float = 0.1
    mem_div_weight: float = 0.5

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
            nn.RMSNorm(dim),
        )
        self.norm = nn.RMSNorm(dim)

    def forward(self, coords: Tensor) -> Tensor:
        if coords.ndim != 3 or coords.shape[-1] != 4:
            raise ValueError(f"expected input of shape (B, N, 4), got {coords.shape}.")

        phase = coords @ self.basis
        fourier = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        pos_enc = self.fourier_project(fourier) + self.coord_project(coords)
        return self.norm(pos_enc)


# ==================================
# Transformer Components
# ==================================


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

        self.attn_norm = nn.RMSNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_project = nn.Linear(dim, dim, bias=False)

        self.ffn_norm = nn.RMSNorm(dim)
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


class CrossTransformerBlock(nn.Module):
    """
    Transformer Block with Cross-Attention.
    Queries attend to keys/values from a different source.

    Input
        q: (B, Nq, D)
        kv: (B, Nkv, D)

    Output
        q_out: (B, Nq, D)
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

        self.q_norm = nn.RMSNorm(dim)
        self.kv_norm = nn.RMSNorm(dim)

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)

        self.out_project = nn.Linear(dim, dim, bias=False)

        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = GEGLU(dim, mult=4)

    def forward(self, q_in: Tensor, kv_in: Tensor) -> Tensor:
        B, Nq, D = q_in.shape
        Nk = kv_in.shape[1]

        q = self.q(self.q_norm(q_in))
        k, v = self.kv(self.kv_norm(kv_in)).chunk(2, dim=-1)

        q = q.view(B, Nq, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Nk, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Nk, self.heads, self.head_dim).transpose(1, 2)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, Nq, D)

        x = q_in + self.out_project(attn_out)

        ffn_in = self.ffn_norm(x)
        ffn_out = self.ffn(ffn_in)

        x = x + ffn_out
        return x


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
        self.norm = nn.RMSNorm(dim)

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


class BottleneckDecoder(nn.Module):
    """
    Bottleneck Decoder with Cross-Attention.

    Input
        vis: (B, N_vis, D)
        coords_enc: (B, N, D)
        mask: (B, N), True = visible

    Output
        pred_masked: (B, N_mask, S)
    """

    def __init__(
        self,
        mem_tokens: int,
        dim: int,
        mem_layers: int,
        query_layers: int,
        heads: int,
        patch_size: int,
    ):
        super().__init__()

        self.mem = nn.Parameter(torch.randn(1, mem_tokens, dim))
        nn.init.orthogonal_(self.mem)

        self.vis_mem = nn.Linear(dim, dim, bias=False)

        self.mask_tok = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_tok, std=0.02)

        self.compress = nn.ModuleList(
            [CrossTransformerBlock(heads, dim) for _ in range(mem_layers)]
        )
        self.mem_attn = nn.ModuleList(
            [TransformerBlock(heads, dim) for _ in range(mem_layers)]
        )

        self.reconstruct = nn.ModuleList(
            [CrossTransformerBlock(heads, dim) for _ in range(query_layers)]
        )

        self.norm = nn.RMSNorm(dim)
        self.predict = nn.Linear(dim, patch_size)

    def forward(self, vis: Tensor, coords_enc: Tensor, mask: Tensor) -> Tensor:
        B, N, D = coords_enc.shape

        masked_coords = coords_enc[~mask].view(B, -1, D)  # (B, N_mask, D)

        vis_summary = vis.mean(dim=1, keepdim=True)  # (B, 1, D)
        vis_mem = self.vis_mem(vis_summary)

        # Stage 1 - visible tokens -> memory tokens
        mem = self.mem.expand(B, -1, -1) + (0.1 * vis_mem)  # (B, mem_tokens, D)
        for compress_blk, mem_attn_blk in zip(self.compress, self.mem_attn):
            mem = compress_blk(mem, vis)  # (B, mem_tokens, D)
            mem, _ = mem_attn_blk(mem)  # (B, mem_tokens, D)

        # Stage 2 - memory tokens + masked coords -> masked queries
        q = (
            self.mask_tok.expand(B, masked_coords.shape[1], D) + masked_coords
        )  # (B, N, D)
        for blk in self.reconstruct:
            q = blk(q, mem)  # (B, N_mask, D)

        q = self.norm(q)
        pred = self.predict(q)  # (B, N_mask, S)

        return pred, mem


# ==================================
# Masking
# ==================================
@torch.no_grad()
def block_mask(
    coords: Tensor,
    mask_ratio: float = 0.55,
    spat_radius: float = 3.0,
    time_radius: float = 3.0,
    n_seeds: int = 6,
) -> Tensor:
    """
    Block-wise Masking Strategy.

    Input
        coords: (B, N, 4)
            Token coordinates in (x, y, z, t) format.
        mask_ratio:
            Fraction of tokens to mask.
        spat_radius:
            Spatial radius for block masking (in same units as x, y, z).
        time_radius:
            Temporal radius for block masking (in same units as t).
        n_seeds:
            Number of random seed blocks per sample.

    Returns
        mask: (B, N), bool
            True for visible tokens, False for masked tokens.
    """
    if coords.ndim != 3 or coords.shape[-1] != 4:
        raise ValueError(f"expected input of shape (B, N, 4), got {coords.shape}.")

    B, N, _ = coords.shape
    device = coords.device

    n_mask = int(mask_ratio * N)
    if n_mask <= 0:
        return torch.ones(B, N, dtype=torch.bool, device=device)
    if n_mask >= N:
        return torch.zeros(B, N, dtype=torch.bool, device=device)

    seeds_idx = torch.randint(0, N, (B, n_seeds), device=device)
    seeds = coords.gather(
        1, seeds_idx.unsqueeze(-1).expand(-1, -1, 4)
    )  # (B, n_seeds, 4)

    xyz = coords[:, :, :3]  # (B, N, 3)
    seeds_xyz = seeds[:, :, :3]  # (B, n_seeds, 3)

    t = coords[:, :, 3]  # (B, N)
    seeds_t = seeds[:, :, 3]  # (B, n_seeds)

    dist_xyz = torch.cdist(seeds_xyz, xyz)  # (B, n_seeds, N)
    dist_t = (seeds_t.unsqueeze(-1) - t.unsqueeze(1)).abs()  # (B, n_seeds, N)

    inside_block = (dist_xyz <= spat_radius) & (
        dist_t <= time_radius
    )  # (B, n_seeds, N)

    score = inside_block.float().sum(dim=1)  # (B, N)
    score = score + torch.rand_like(score) * 1e-6  # small noise to break ties

    masked_idx = torch.topk(score, n_mask, dim=1).indices  # (B, n_mask)

    masked = torch.zeros(B, N, dtype=torch.bool, device=device)
    masked = masked.scatter_(1, masked_idx, True)

    mask = ~masked
    return mask


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
        self.dec = BottleneckDecoder(
            cfg.mem_tokens,
            cfg.dim,
            cfg.mem_layers,
            cfg.query_layers,
            cfg.dec_heads,
            cfg.patch_size,
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
            raise ValueError(f"expected input of shape (B, C, 3), got {xyz.shape}.")

        B, C, _ = xyz.shape
        device = xyz.device

        t = torch.arange(P, device=device, dtype=xyz.dtype)  # (P,)
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
        pat = pat.flatten(1, 2)  # (B, N, S)
        return tok, pat

    def _normalize_pats(self, pat: Tensor) -> Tensor:
        """
        Normalize patches by subtracting mean over patch dimension.

        Input
            pat: (B, N, S)

        Output
            pat_norm: (B, N, S)
        """
        mean = pat.mean(dim=-1, keepdim=True)  # (B, N, 1)
        std = pat.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (pat - mean) / std

    def _gather_vis(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Gather visible tokens based on mask.

        Input
            x: (B, N, D)
            mask: (B, N), bool

        Output
            vis: (B, N_vis, D)
        """
        B, _, D = x.shape
        return x[mask].view(B, -1, D)

    def _gather_mask(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Gather masked tokens based on mask.

        Input
            x: (B, N, D)
            mask: (B, N), bool

        Output
            masked: (B, N_mask, D)
        """
        B, _, D = x.shape
        return x[~mask].view(B, -1, D)

    def _compute_mem_div_loss(self, mem: Tensor) -> Tensor:
        """
        Compute diversity loss on memory tokens to encourage them to learn different aspects of the data.

        Input
            mem: (B, mem_tokens, D)

        Output
            loss: scalar Tensor
        """
        mem = torch.nn.functional.normalize(mem, dim=-1)  # (B, mem_tokens, D)
        sim = mem @ mem.transpose(1, 2)  # (B, mem_tokens, mem_tokens)

        mem_tokens = mem.shape[1]
        eye = torch.eye(mem_tokens, dtype=torch.bool, device=sim.device).unsqueeze(0)
        off_diag = sim.masked_select(~eye)

        loss = (off_diag**2).mean()
        return loss

    def forward(self, x: Tensor, xyz: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected input of shape (B, C, T), got {x.shape}.")

        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"expected input of shape (B, C, 3), got {xyz.shape}.")

        if x.shape[:2] != xyz.shape[:2]:
            raise ValueError(
                f"batch and channel dimensions of x and xyz must match, got {x.shape} and {xyz.shape}."
            )

        tok, pat = self.patch(x)  # (B, C, P, D), (B, C, P, S)
        P = tok.shape[2]

        tok, targets = self._flatten_tokens(tok, pat)  # (B, N, D), (B, N, S)

        coords = self._build_coords(xyz, P)  # (B, N, 4)
        coords_enc = self.coord_enc(coords)  # (B, N, D)

        tok = tok + coords_enc  # (B, N_vis, D)
        z, _ = self.enc(tok)  # (B, N_vis, D), List[(B, N_vis, D)]

        return z
