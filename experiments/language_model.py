"""The paper's plainLM transformer; initialization and arithmetic preserved."""
import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor


@dataclasses.dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    dim: int
    expand: float
    n_layers: int
    n_heads: int
    mlp: str = "glu"
    rmsnorm_eps: float = 1e-6
    tie_embeddings: bool = False
    qk_norm: bool = True
    embed_norm: bool = True
    rope_theta: float = 500000.0

    @property
    def block_size(self) -> int:
        return self.seq_len

    @property
    def n_embd(self) -> int:
        return self.dim

    @property
    def n_layer(self) -> int:
        return self.n_layers

    @property
    def n_head(self) -> int:
        return self.n_heads

    def to_dict(self):
        data = dataclasses.asdict(self)
        data.update(
            block_size=self.seq_len,
            n_embd=self.dim,
            n_layer=self.n_layers,
            n_head=self.n_heads,
        )
        return data


def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 10000.0,
        condense_ratio: int = 1,
):
    inv_freqs = 1.0 / (
        theta ** (
            torch.arange(
                0, dim, 2,
                dtype=torch.float32,
                device=torch.device("cpu"),
            ) / dim
        )
    )
    t = torch.arange(
        end, dtype=torch.float32,
        device=inv_freqs.device,
    ) / condense_ratio
    freqs = torch.outer(t, inv_freqs).float()
    return torch.stack(
        [
            torch.cos(freqs)[None, :, None, :],
            torch.sin(freqs)[None, :, None, :],
        ],
        dim=4,
    )


def apply_rotary_emb_complex_like(
        q: torch.Tensor,
        k: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> tuple[Tensor, ...]:
    qk_r2 = torch.cat([q, k], dim=2).unflatten(
        dim=-1,
        sizes=(-1, 2),
    ).float()
    rotated_qk = torch.stack(
        [
            qk_r2[..., 0] * freqs_cis[..., 0]
            - qk_r2[..., 1] * freqs_cis[..., 1],
            qk_r2[..., 1] * freqs_cis[..., 0]
            + qk_r2[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    ).flatten(3)
    return torch.split(rotated_qk.type_as(q), q.shape[2], dim=2)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class GLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x, z = self.fc1(x).split(self.hidden_dim, dim=2)
        return self.fc2(F.silu(x) * z)


class MLPReluSquared(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)).pow(2))


MLP_CLASSES = {
    "mlp": MLP,
    "glu": GLU,
    "mlp_relu_sq": MLPReluSquared,
}


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.dim % cfg.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.w_qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.w_out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.q_norm = (
            RMSNorm(self.head_dim, cfg.rmsnorm_eps)
            if cfg.qk_norm else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(self.head_dim, cfg.rmsnorm_eps)
            if cfg.qk_norm else nn.Identity()
        )

    def forward(self, x, freqs_cis, attn_mask=None):
        bsz, seqlen, dim = x.shape
        q, k, v = self.w_qkv(x).split(dim, dim=2)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if attn_mask is not None:
            if attn_mask.ndim == 3:
                attn_mask = attn_mask.unsqueeze(1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.w_out(out)


class Block(nn.Module):
    def __init__(self, layer_id: int, cfg: ModelConfig):
        super().__init__()
        self.attn = Attention(cfg)
        self.attn_norm = RMSNorm(cfg.dim, cfg.rmsnorm_eps)
        self.mlp = MLP_CLASSES[cfg.mlp](
            dim=cfg.dim,
            hidden_dim=int(cfg.expand * cfg.dim),
        )
        self.mlp_norm = RMSNorm(cfg.dim, cfg.rmsnorm_eps)
        self.layer_id = layer_id

    def forward(self, x, freqs_cis, attn_mask):
        x = x + self.attn(self.attn_norm(x), freqs_cis, attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.n_layers = cfg.n_layers
        head_dim = cfg.dim // cfg.n_heads
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.embed_norm = (
            RMSNorm(cfg.dim, cfg.rmsnorm_eps)
            if cfg.embed_norm else nn.Identity()
        )
        self.layers = nn.ModuleList([
            Block(layer_id, cfg) for layer_id in range(cfg.n_layers)
        ])
        self.out_norm = RMSNorm(cfg.dim, cfg.rmsnorm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, cfg.seq_len, cfg.rope_theta)[
                :, :cfg.seq_len
            ],
            persistent=False,
        )
        self.apply(self._init_weights)
        self._scale_residual_branches()
        if cfg.tie_embeddings:
            self.tie_weights()

    def forward(self, x, targets=None, attn_mask=None):
        seqlen = x.shape[1]
        h = self.embed_tokens(x)
        h = self.embed_norm(h)
        freqs_cis = self.freqs_cis[:, :seqlen].to(h.device)
        for layer in self.layers:
            h = layer(h, freqs_cis, attn_mask)
        logits = self.lm_head(self.out_norm(h))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_branches(self):
        for name, param in self.named_parameters():
            if name.endswith("fc2.weight"):
                torch.nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * self.n_layers),
                )
            if name.endswith("w_out.weight"):
                torch.nn.init.normal_(
                    param,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * self.n_layers),
                )

    def tie_weights(self):
        self.lm_head.weight = self.embed_tokens.weight

    def count_params(self, non_embedding=True):
        n_params = sum(param.numel() for param in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
            if self.lm_head.weight is not self.embed_tokens.weight:
                n_params -= self.lm_head.weight.numel()
        return n_params

    def get_num_params(self, non_embedding=True):
        return self.count_params(non_embedding=non_embedding)
