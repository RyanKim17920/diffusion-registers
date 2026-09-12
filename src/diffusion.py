"""Data loading, the absorbing-state masking process, loss, and the
one-token-per-step denoising decoder."""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import MASK, N_CELLS, build_tokens


# ---------------------------------------------------------------- data

def load_split(data_dir, split):
    d = os.path.join(data_dir, split)
    out = {
        "puzzles": np.load(os.path.join(d, "puzzles.npy")),
        "solutions": np.load(os.path.join(d, "solutions.npy")),
        "n_clues": np.load(os.path.join(d, "n_clues.npy")),
    }
    n = out["puzzles"].shape[0]
    assert out["solutions"].shape == (n, N_CELLS)
    assert out["n_clues"].shape == (n,)
    return out


def read_meta(data_dir):
    p = os.path.join(data_dir, "meta.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


# ------------------------------------------------- masking / diffusion

def sample_masks(n, rng, min_masked=1):
    """Absorbing-state forward process.

    Draw t ~ U(0, 1] per example and mask each of the 81 solution positions
    independently with probability t. At least `min_masked` positions are
    always masked, so every example contributes a loss term.

    Returns (t (n,), mask (n, 81) bool) as numpy.
    """
    t = rng.random(n).astype(np.float32)
    t = np.clip(t, 1.0 / N_CELLS, 1.0)
    mask = rng.random((n, N_CELLS)).astype(np.float32) < t[:, None]
    # force at least min_masked positions masked
    short = mask.sum(1) < min_masked
    if short.any():
        idx = np.where(short)[0]
        for i in idx:
            pick = rng.choice(N_CELLS, size=min_masked, replace=False)
            mask[i, pick] = True
    return t, mask


def make_inputs(puzzles, solutions, mask, device):
    """Build model input tokens: masked positions -> MASK, others -> the true
    digit (teacher-forced partial reveal)."""
    p = torch.as_tensor(puzzles, device=device, dtype=torch.long)
    s = torch.as_tensor(solutions, device=device, dtype=torch.long)
    m = torch.as_tensor(mask, device=device)
    sol_tok = torch.where(m, torch.full_like(s, MASK), s)
    return build_tokens(p, sol_tok), s, m


def denoising_loss(logits, solutions, mask, t=None):
    """Cross-entropy on masked solution positions only.

    Returns (mean_ce, elbo_ce). `mean_ce` is CE averaged over masked positions
    -- the low-variance quantity we optimise and report. `elbo_ce` is the MDLM
    weighting (1/t) * sum_masked CE / 81, reported for reference. Registers
    never appear here: logits cover only the 81 solution positions.
    """
    tgt = solutions - 1  # digits 1..9 -> classes 0..8
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none"
    ).view_as(tgt)
    m = mask.to(ce.dtype)
    denom = m.sum().clamp(min=1.0)
    mean_ce = (ce * m).sum() / denom
    if t is None:
        return mean_ce, None
    per_ex = (ce * m).sum(1) / N_CELLS
    elbo = (per_ex / t).mean()
    return mean_ce, elbo


# ------------------------------------------------------ decoding / eval

@torch.no_grad()
def solve(model, puzzles, device, reg_mode="normal", reveal_per_step=1,
          chunk=1024, amp_dtype=torch.bfloat16):
    """Confidence-ordered denoising decoder.

    Starts from all 81 solution positions MASKed, and at each step reveals the
    `reveal_per_step` masked positions whose argmax probability is highest.
    Returns the decoded grids as (N, 81) int64 on CPU, values 1..9.
    """
    model.eval()
    outs = []
    for i in range(0, len(puzzles), chunk):
        p = torch.as_tensor(puzzles[i:i + chunk], device=device, dtype=torch.long)
        B = p.shape[0]
        sol = torch.full((B, N_CELLS), MASK, dtype=torch.long, device=device)
        remaining = N_CELLS
        while remaining > 0:
            k = min(reveal_per_step, remaining)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(build_tokens(p, sol), reg_mode=reg_mode)
            probs = logits.float().softmax(-1)
            conf, digit = probs.max(-1)  # (B, 81)
            conf = conf.masked_fill(sol != MASK, -1.0)
            pick = conf.topk(k, dim=-1).indices  # (B, k)
            vals = digit.gather(1, pick) + 1
            sol.scatter_(1, pick, vals)
            remaining -= k
        outs.append(sol.cpu())
    return torch.cat(outs, 0)


def solve_metrics(decoded, solutions, n_clues=None):
    """Exact-solve accuracy and per-cell accuracy, optionally bucketed by
    clue count."""
    sol = torch.as_tensor(np.ascontiguousarray(solutions), dtype=torch.long)
    correct = decoded == sol
    exact = correct.all(1)
    out = {
        "exact_solve_acc": exact.float().mean().item(),
        "cell_acc": correct.float().mean().item(),
        "n": int(len(exact)),
    }
    if n_clues is not None:
        nc = torch.as_tensor(np.ascontiguousarray(n_clues), dtype=torch.long)
        by = {}
        for c in sorted(set(nc.tolist())):
            sel = nc == c
            by[int(c)] = {
                "exact_solve_acc": exact[sel].float().mean().item(),
                "cell_acc": correct[sel].float().mean().item(),
                "n": int(sel.sum()),
            }
        out["by_clues"] = by
    return out
