"""MuonAdam and Adam as pure functions on MLX trees, ported from XMimic's optim.py.

Muon (torch.optim.Muon with match_rms_adamw) runs on leaves where ``mask`` is True, torch-style Adam on the rest.
Newton-Schulz runs in float32 here (bf16 in the JAX and torch versions).
"""

from __future__ import annotations

import mlx.core as mx

from .nets import leaves, tmap

NS_COEFFS = (3.4445, -4.775, 2.0315)


def newton_schulz(G, steps: int = 5, eps: float = 1e-7):
    """Quintic Newton-Schulz orthogonalization as in torch.optim's Muon, on the wide orientation of G."""
    a, b, c = NS_COEFFS
    X = G
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    X = X / mx.maximum(mx.sqrt((X * X).sum()), eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if tall else X


def init(params, mask):
    def leaf(p, m):
        return {"buf": mx.zeros_like(p)} if m else {"m": mx.zeros_like(p), "v": mx.zeros_like(p)}

    return {"leaves": tmap(leaf, params, mask), "t": mx.zeros((), mx.int32)}


def clip_by_global_norm(grads, max_norm: float):
    """Like torch's clip_grad_norm_. Returns (clipped grads, total norm before clipping)."""
    norm = mx.sqrt(sum((g * g).sum() for g in leaves(grads)))
    coef = mx.minimum(max_norm / (norm + 1e-6), 1.0)
    return tmap(lambda g: g * coef, grads), norm


def step(params, grads, state, mask, lr, betas=(0.9, 0.999), eps: float = 1e-8, momentum: float = 0.95, ok=None):
    """One MuonAdam step, plain Adam if mask is all False.

    ``ok`` (0-dim bool) gates the step: when False, params and optimizer state come back unchanged.
    """
    b1, b2 = betas
    t = state["t"] + 1
    tf = t.astype(mx.float32)
    bc1, bc2 = 1 - mx.power(b1, tf), 1 - mx.power(b2, tf)

    def new_p(p, g, s, m):
        if m:
            buf = momentum * s["buf"] + (1 - momentum) * g
            upd = newton_schulz(g + momentum * (buf - g))
            ratio = 0.2 * max(p.shape[0], p.shape[1]) ** 0.5
            return p - lr * ratio * upd
        mm = b1 * s["m"] + (1 - b1) * g
        vv = b2 * s["v"] + (1 - b2) * g * g
        return p - (lr / bc1) * mm / (mx.sqrt(vv) / mx.sqrt(bc2) + eps)

    def new_s(g, s, m):
        if m:
            return {"buf": momentum * s["buf"] + (1 - momentum) * g}
        return {"m": b1 * s["m"] + (1 - b1) * g, "v": b2 * s["v"] + (1 - b2) * g * g}

    # optimizer state has an extra dict per leaf, so recurse on the parameter tree
    def walk(fn, p, *rest):
        if isinstance(p, dict):
            return {k: walk(fn, p[k], *(r[k] for r in rest)) for k in p}
        if isinstance(p, (list, tuple)):
            return type(p)(walk(fn, *xs) for xs in zip(p, *rest))
        return fn(p, *rest)

    out_p = walk(new_p, params, grads, state["leaves"], mask)
    out_s = {"leaves": walk(lambda p, g, s, m: new_s(g, s, m), params, grads, state["leaves"], mask), "t": t}
    if ok is None:
        return out_p, out_s
    keep = lambda new, old: tmap(lambda a, b: mx.where(ok, a, b), new, old)  # noqa: E731
    return keep(out_p, params), keep(out_s, state)


def ema(avg, live, decay: float, first: bool):
    """Weight EMA as in torch's AveragedModel: the first update copies, later ones lerp toward the live weights."""
    if first:
        return tmap(lambda a, live_: live_, avg, live)
    return tmap(lambda a, live_: a + (1 - decay) * (live_ - a), avg, live)
