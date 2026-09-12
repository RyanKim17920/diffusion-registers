"""Masked discrete diffusion Transformer with loss-free register tokens.

Sequence layout (bidirectional attention, no causal mask):

    [ puzzle: 81 tokens ] [ SEP ] [ solution: 81 tokens ] [ R_0 .. R_{K-1} ]
      0..80                81      82..162                 163..163+K-1

Token vocabulary (12 ids):
    0       BLANK   an empty cell in the puzzle region
    1..9    digits
    10      MASK    an un-revealed solution position
    11      SEP

Registers are NOT vocabulary entries. They are K rows of a dedicated learned
embedding table that is re-read on every forward pass, so register state is
never carried across denoising steps -- each denoising step starts from the
same learned initialisation and whatever the registers "compute" lives only
inside that one forward pass.

Registers have no prediction target and no loss term: the output head is only
applied to the 81 solution positions. Registers influence the loss only
through attention into the solution positions.
"""

import json
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

BLANK = 0
MASK = 10
SEP = 11
VOCAB_SIZE = 12

N_CELLS = 81
PUZZLE_START = 0
SEP_POS = 81
SOL_START = 82
SEQ_REAL = 163  # 81 + 1 + 81
N_DIGITS = 9  # prediction classes: digit d -> class d-1


@dataclass
class ModelConfig:
    n_registers: int = 0
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.0
    # Sequence shape. The defaults describe the Sudoku task, so model configs
    # written before these fields existed still load unchanged.
    vocab_size: int = VOCAB_SIZE
    seq_len: int = SEQ_REAL
    out_start: int = SOL_START     # first position the head is applied to
    out_len: int = N_CELLS         # number of predicted positions
    n_classes: int = N_DIGITS      # prediction classes at those positions
    tie_head: bool = False         # tie the output head to the token embedding

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            return ModelConfig(**json.load(f))


class Block(nn.Module):
    """Pre-LN transformer block with an optional explicit-attention path.

    The fast path uses fused SDPA. When `collect` is a list we instead compute
    attention weights explicitly so they can be inspected -- slower, used only
    by the analysis code.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        assert self.d_head * cfg.n_heads == cfg.d_model
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, collect=None):
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each (B, nh, T, dh)
        if collect is None:
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            att = att.softmax(dim=-1)
            collect.append(att.detach())
            y = att @ v
        y = y.transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(y))
        h = self.ln2(x)
        x = x + self.drop(self.fc2(F.gelu(self.fc1(h))))
        return x


class RegisterDiffusionTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.n_registers
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(cfg.seq_len, cfg.d_model))
        # Register embeddings double as the registers' content and position
        # signal. Re-read every forward pass -> stateless across steps.
        if self.K > 0:
            self.reg_emb = nn.Parameter(torch.zeros(self.K, cfg.d_model))
        else:
            self.register_parameter("reg_emb", None)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.n_classes, bias=not cfg.tie_head)
        self.apply(self._init)
        if cfg.tie_head:
            assert cfg.n_classes == cfg.vocab_size, \
                "tie_head needs the head to span the vocabulary"
            self.head.weight = self.tok_emb.weight
        nn.init.normal_(self.pos_emb, std=0.02)
        if self.K > 0:
            nn.init.normal_(self.reg_emb, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def _registers(self, B, device, reg_mode, generator=None, reg_state=None):
        """Build the (B, K, D) register block fed into this forward pass.

        reg_state is the carried state from the previous denoising step within
        the same generation block. When it is None the registers are
        (re-)initialised from the learned embeddings -- that happens at the
        start of every generation block, never mid-block.

        reg_mode:
          'normal'  -- use the registers as they are
          'zero'    -- all-zero registers (ablation: removes their content)
          'shuffle' -- per-example random permutation across the K slots
                       (ablation: keeps the content, destroys slot identity)
        """
        if self.K == 0:
            return None
        if reg_state is None:
            reg = self.reg_emb.unsqueeze(0).expand(B, -1, -1)
        else:
            reg = reg_state
        if reg_mode == "normal":
            return reg
        if reg_mode == "zero":
            return torch.zeros_like(reg)
        if reg_mode == "shuffle":
            idx = torch.argsort(
                torch.rand(B, self.K, device=device, generator=generator), dim=-1
            )
            return torch.gather(
                reg, 1, idx.unsqueeze(-1).expand(-1, -1, reg.shape[-1])
            )
        raise ValueError(f"unknown reg_mode {reg_mode!r}")

    def forward(self, tokens, reg_mode="normal", collect_attn=False,
                generator=None, return_hidden=False, reg_state=None,
                return_reg_state=False):
        """tokens: (B, 163) long. Returns logits (B, 81, 9) for solution cells.

        If collect_attn, also returns the per-layer attention weights
        (list of (B, n_heads, T, T)).
        """
        B, T = tokens.shape
        assert T == self.cfg.seq_len, \
            f"expected {self.cfg.seq_len} tokens, got {T}"
        x = self.tok_emb(tokens) + self.pos_emb.unsqueeze(0)
        reg = self._registers(B, tokens.device, reg_mode, generator, reg_state)
        if reg is not None:
            x = torch.cat([x, reg.to(x.dtype)], dim=1)
        collect = [] if collect_attn else None
        for blk in self.blocks:
            x = blk(x, collect=collect)
        h = self.ln_f(x)
        c = self.cfg
        # the carried state is the normalised final hidden state at the
        # register positions; ln_f keeps it from drifting in scale as it is
        # passed from one denoising step to the next
        new_reg = h[:, self.cfg.seq_len:, :] if self.K else None
        out = h[:, c.out_start:c.out_start + c.out_len, :]
        # return_hidden lets the caller apply the head to a gathered subset of
        # positions; at a 50k vocabulary, materialising logits everywhere is
        # what blows up memory, not the batch size.
        if not return_hidden:
            out = self.head(out)
        if collect_attn:
            return (out, new_reg, collect) if return_reg_state else (out, collect)
        return (out, new_reg) if return_reg_state else out

    @torch.no_grad()
    def hidden_states(self, tokens, reg_mode="normal"):
        """Per-layer hidden states (list of (B, T, D)), for norm analysis."""
        B, T = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb.unsqueeze(0)
        reg = self._registers(B, tokens.device, reg_mode)
        if reg is not None:
            x = torch.cat([x, reg.to(x.dtype)], dim=1)
        out = [x]
        for blk in self.blocks:
            x = blk(x)
            out.append(x)
        return out


def build_tokens(puzzles, sol_tokens):
    """puzzles: (B, 81) uint8/long in 0..9. sol_tokens: (B, 81) long in
    {1..9, MASK}. Returns (B, 163) long."""
    B = puzzles.shape[0]
    sep = torch.full((B, 1), SEP, dtype=torch.long, device=puzzles.device)
    return torch.cat([puzzles.long(), sep, sol_tokens.long()], dim=1)
