"""Semi-autoregressive block diffusion with carried register state.

The 81 solution cells are split into blocks of `block_size` consecutive cells.
Generation proceeds block by block; within a block the model reveals one cell
per denoising step, and the register state is CARRIED from step to step. At
each block boundary the registers are re-initialised from the learned
embeddings.

So a register's lifetime is exactly one generation block: it is written and
read across the steps of that block and then discarded. Registers still have
no prediction target and no direct loss -- the only path to them is the
gradient that flows back through the carry from the token predictions of later
steps in the same block.

Training mirrors this exactly. For each example we sample a block index b:

    blocks < b : revealed, teacher-forced to the true digits
    block b    : the one being generated, revealed one cell per step in a
                 random order, register state carried across those steps
    blocks > b : fully masked context

and the loss at step j is the cross-entropy over the cells of block b that are
still masked at step j. The whole block is unrolled with gradient, so the
model is trained to write something into the registers at step j that is worth
reading at step j+1.
"""

import numpy as np
import torch
import torch.nn.functional as F

from model import MASK, N_CELLS, build_tokens


def n_blocks_for(block_size):
    assert N_CELLS % block_size == 0, \
        f"block_size must divide {N_CELLS}, got {block_size}"
    return N_CELLS // block_size


def sample_block_plan(n, block_size, rng):
    """Per example: which block is being generated, and in what order its
    cells are revealed.

    Returns (b (n,), order (n, block_size)) where `order` holds absolute cell
    indices in reveal order.
    """
    nb = n_blocks_for(block_size)
    b = rng.integers(0, nb, size=n)
    perm = np.argsort(rng.random((n, block_size)), axis=1)
    order = b[:, None] * block_size + perm
    return b, order


def initial_sol_tokens(solutions, b, block_size, device):
    """Solution-region tokens at the start of block b: earlier blocks
    teacher-forced to the truth, block b and everything after masked."""
    s = torch.as_tensor(np.ascontiguousarray(solutions), device=device,
                        dtype=torch.long)
    cell_block = (torch.arange(N_CELLS, device=device) // block_size)
    bt = torch.as_tensor(b, device=device, dtype=torch.long)[:, None]
    revealed = cell_block[None, :] < bt
    return torch.where(revealed, s, torch.full_like(s, MASK)), s


def unrolled_block_loss(model, puzzles, solutions, b, order, block_size,
                        device, reg_mode="normal"):
    """Unroll one generation block with register carry and accumulate the loss.

    Returns (mean_ce, n_terms). Gradient flows through the carried register
    state, which is the only way the registers can learn anything.
    """
    p = torch.as_tensor(np.ascontiguousarray(puzzles), device=device,
                        dtype=torch.long)
    sol_tok, truth = initial_sol_tokens(solutions, b, block_size, device)
    ordt = torch.as_tensor(order, device=device, dtype=torch.long)

    reg_state = None
    ce_sum, n_terms = None, 0
    for j in range(block_size):
        tokens = build_tokens(p, sol_tok)
        out = model(tokens, reg_mode=reg_mode, reg_state=reg_state,
                    return_reg_state=True)
        logits, reg_state = out
        # loss over the cells of this block still masked at step j
        tgt_idx = ordt[:, j:]                                  # (B, n_left)
        lg = torch.gather(
            logits, 1,
            tgt_idx.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
        )
        tgt = torch.gather(truth, 1, tgt_idx) - 1
        part = F.cross_entropy(
            lg.reshape(-1, lg.shape[-1]).float(), tgt.reshape(-1),
            reduction="sum",
        )
        ce_sum = part if ce_sum is None else ce_sum + part
        n_terms += tgt.numel()
        # reveal the j-th cell of the block, teacher-forced to the truth
        reveal = ordt[:, j:j + 1]
        sol_tok = sol_tok.scatter(1, reveal, torch.gather(truth, 1, reveal))
    return ce_sum / max(n_terms, 1), n_terms


@torch.no_grad()
def block_solve(model, puzzles, device, block_size, reg_mode="normal",
                chunk=512, amp_dtype=torch.bfloat16):
    """Block-diffusion decoding.

    Blocks are generated in order. Within a block the highest-confidence
    masked cell OF THAT BLOCK is revealed each step, and the register state is
    carried across the block's steps, then reset at the block boundary.
    """
    model.eval()
    nb = n_blocks_for(block_size)
    outs = []
    for i in range(0, len(puzzles), chunk):
        p = torch.as_tensor(np.ascontiguousarray(puzzles[i:i + chunk]),
                            device=device, dtype=torch.long)
        B = p.shape[0]
        sol = torch.full((B, N_CELLS), MASK, dtype=torch.long, device=device)
        cell_block = torch.arange(N_CELLS, device=device) // block_size
        for b in range(nb):
            in_block = (cell_block == b)[None, :].expand(B, -1)
            reg_state = None
            for _ in range(block_size):
                with torch.autocast("cuda", dtype=amp_dtype,
                                    enabled=device.type == "cuda"):
                    out = model(build_tokens(p, sol), reg_mode=reg_mode,
                                reg_state=reg_state, return_reg_state=True)
                logits, reg_state = out
                probs = logits.float().softmax(-1)
                conf, digit = probs.max(-1)
                # only cells of the current block are eligible
                conf = conf.masked_fill(~in_block | (sol != MASK), -1.0)
                pick = conf.argmax(-1, keepdim=True)
                sol.scatter_(1, pick, torch.gather(digit, 1, pick) + 1)
        outs.append(sol.cpu())
    return torch.cat(outs, 0)
