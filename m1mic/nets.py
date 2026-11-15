"""World model, policy (motion encoder + shared decoder) and critic as pure functions on parameter trees.

MLX port of XMimic's nets.py, float32 throughout. Running normalizers live in separate buffer trees that are updated
from data, not by gradients.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .mdp import invariant_features
from .rot import mm3, mv3, so3_exp, so3_log, tr3, yaw_mat

silu = nn.silu


def elu(x):
    """ELU with a finite gradient everywhere, like jax.nn.elu.

    mlx.nn.elu is where(x > 0, x, exp(x) - 1). Above x = 88 the unselected branch overflows and the gradient is NaN
    while the value stays finite, which sooner or later puts NaNs in the critic and then in the policy."""
    return mx.where(x > 0, x, mx.expm1(mx.where(x > 0, 0.0, x)))


# ---------------------------------------------------------------- trees (dicts, lists and tuples of arrays)
def tmap(fn, *trees):
    """Apply fn leaf-wise to one or more trees of the same structure."""
    t = trees[0]
    if isinstance(t, dict):
        return {k: tmap(fn, *(x[k] for x in trees)) for k in t}
    if isinstance(t, (list, tuple)):
        return type(t)(tmap(fn, *xs) for xs in zip(*trees))
    return fn(*trees)


def leaves(tree):
    if isinstance(tree, dict):
        return [x for v in tree.values() for x in leaves(v)]
    if isinstance(tree, (list, tuple)):
        return [x for v in tree for x in leaves(v)]
    return [tree]


# ---------------------------------------------------------------- MLPs
def mlp_init(rng: np.random.Generator, sizes: list[int], layer_norm: bool = False):
    """Linear layers with torch.nn.Linear's default init, U(+-1/sqrt(fan_in)). Weights are stored (in, out)."""
    layers = []
    for i, o in zip(sizes[:-1], sizes[1:]):
        bound = 1.0 / np.sqrt(i)
        layers.append({"w": mx.array(rng.uniform(-bound, bound, (i, o)).astype(np.float32)),
                       "b": mx.array(rng.uniform(-bound, bound, (o,)).astype(np.float32))})  # fmt: skip
    if layer_norm:
        for layer in layers[:-1]:
            n = layer["b"].shape[0]
            layer["ln_g"], layer["ln_b"] = mx.ones(n), mx.zeros(n)
    return layers


def mlp_apply(layers, x, act):
    """Linear -> [LayerNorm] -> activation for every layer but the last."""
    h = x
    last = len(layers) - 1
    for i, layer in enumerate(layers):
        h = h @ layer["w"] + layer["b"]
        if i == last:
            return h
        if "ln_g" in layer:
            h = mx.fast.layer_norm(h, layer["ln_g"], layer["ln_b"], 1e-5)
        h = act(h)


def muon_mask(layers):
    """True for the hidden weight matrices (every Linear except the first and last), as in BlinkMimic's MuonAdam."""
    return [{k: (k == "w" and 0 < i < len(layers) - 1) for k in layer} for i, layer in enumerate(layers)]


# ---------------------------------------------------------------- running normalizer
def rn_init(dim: int):
    return {"mean": mx.zeros(dim), "var": mx.ones(dim), "count": mx.zeros(())}


def rn_update(st, x, max_count: float = 2e8):
    """Merge the batch statistics until max_count samples, then freeze (BlinkMimic's RunningNorm)."""
    x = x.reshape(-1, x.shape[-1])
    n = x.shape[0]
    m, v = x.mean(0), x.var(0)
    cnt = st["count"]
    tot = cnt + n
    delta = m - st["mean"]
    mean = st["mean"] + delta * n / tot
    var = (st["var"] * cnt + v * n + delta**2 * cnt * n / tot) / tot
    live = cnt < max_count
    return {"mean": mx.where(live, mean, st["mean"]), "var": mx.where(live, var, st["var"]), "count": mx.where(live, tot, cnt)}


def rn_apply(st, x, eps: float = 1e-2):
    return (x - st["mean"]) / mx.sqrt(st["var"] + eps**2)


def rn_std(st, eps: float = 1e-2):
    return mx.sqrt(st["var"] + eps**2)


# ---------------------------------------------------------------- world model (semi-implicit, BlinkMimic --wm_structured)
# Predicts normalized state increments in the root (heading) frame. Positions get a semi-implicit drift from the old
# and new velocities, so the network only predicts the residual.
DRIFT_W = 0.625  # (n + 1) / (2 n) for n = 4 semi-implicit substeps


def wm_init(rng, in_dim: int, out_dim: int, hidden=(1024, 1024, 1024)):
    layers = mlp_init(rng, [in_dim, *hidden, out_dim], layer_norm=True)
    layers[-1]["w"] = mx.zeros_like(layers[-1]["w"])
    layers[-1]["b"] = mx.zeros_like(layers[-1]["b"])
    return {"net": layers}, {"in_norm": rn_init(in_dim), "out_norm": rn_init(out_dim), "inc_norm": rn_init(out_dim)}


def wm_inputs(S, a, xi_n):
    return mx.concatenate([invariant_features(S), a, xi_n], axis=-1)


def _drift(S, v1, w1, qd1, dt):
    p, R, v, w, q, qd = S
    a, b = dt * (1 - DRIFT_W), dt * DRIFT_W
    return mv3(tr3(yaw_mat(R)), a * v + b * v1), mv3(tr3(R), a * w + b * w1), a * qd + b * qd1


def increments(S, S1):
    p, R, v, w, q, qd = S
    p1, R1, v1, w1, q1, qd1 = S1
    Rt = tr3(R)
    return mx.concatenate([mv3(tr3(yaw_mat(R)), p1 - p), so3_log(mm3(Rt, R1)), mv3(Rt, v1 - v), mv3(Rt, w1 - w), q1 - q, qd1 - qd], axis=-1)


def _split(d, J):
    return d[..., 0:3], d[..., 3:6], d[..., 6:9], d[..., 9:12], d[..., 12 : 12 + J], d[..., 12 + J :]


def wm_targets(S, S1, dt):
    inc = increments(S, S1)
    dp, dth, dv, dw, dq, dqd = _split(inc, S[4].shape[-1])
    gp, gth, gq = _drift(S, S1[2], S1[3], S1[5], dt)
    return mx.concatenate([dp - gp, dth - gth, dv, dw, dq - gq, dqd], axis=-1)


def wm_integrate(S, d, dt):
    p, R, v, w, q, qd = S
    dp, dth, dv, dw, dq, dqd = _split(d, q.shape[-1])
    v1, w1, qd1 = v + mv3(R, dv), w + mv3(R, dw), qd + dqd
    gp, gth, gq = _drift(S, v1, w1, qd1, dt)
    return p + mv3(yaw_mat(R), dp + gp), mm3(R, so3_exp(dth + gth)), v1, w1, q + dq + gq, qd1


def wm_step(params, bufs, S, a, xi_n, dt):
    """One imagined control step: (state, action, normalized sim context) -> next state."""
    y = mlp_apply(params["net"], rn_apply(bufs["in_norm"], wm_inputs(S, a, xi_n)), silu)
    return wm_integrate(S, y * rn_std(bufs["out_norm"], 1e-4) + bufs["out_norm"]["mean"], dt)


def state_error(S_hat, S):
    """Prediction error in the true state's root frame (the world-model loss before normalization)."""
    p, R, v, w, q, qd = S
    ph, Rh, vh, wh, qh, qdh = S_hat
    Rt = tr3(R)
    return mx.concatenate([mv3(tr3(yaw_mat(R)), ph - p), so3_log(mm3(Rt, Rh)), mv3(Rt, vh - v), mv3(Rt, wh - w), qh - q, qdh - qd], axis=-1)


# ---------------------------------------------------------------- policy
VEL_DIM = 3  # velocity command: (vx, vy, yaw rate) in the body frame


def pi_init(rng, obs_dim: int, goal_dim: int, act_dim: int, hist: int, z_dim: int = 32,
            enc_hidden=(2048, 1024, 512, 256), dec_hidden=(2048, 1024, 512, 256), vel: bool = False,
            vel_hidden=(256, 256)):  # fmt: skip
    """Motion encoder ``enc`` and shared decoder ``dec``. With ``vel``, also the velocity encoder ``enc_vel``
    (command and the last ``hist`` observations -> z) and its command normalizer ``goal_norm_vel``."""
    enc = mlp_init(rng, [goal_dim, *enc_hidden, z_dim])
    dec = mlp_init(rng, [z_dim + obs_dim * hist, *dec_hidden, act_dim])
    dec[-1]["w"] = dec[-1]["w"] * 0.01
    dec[-1]["b"] = mx.zeros_like(dec[-1]["b"])
    params, bufs = {"enc": enc, "dec": dec}, {"obs_norm": rn_init(obs_dim), "goal_norm": rn_init(goal_dim)}
    if vel:
        params["enc_vel"] = mlp_init(rng, [VEL_DIM + obs_dim * hist, *vel_hidden, z_dim])
        bufs["goal_norm_vel"] = rn_init(VEL_DIM)
    return params, bufs


def pi_encode(params, bufs, goal):
    return mx.tanh(mlp_apply(params["enc"], rn_apply(bufs["goal_norm"], goal), elu))


def obs_hist_n(bufs, obs_hist):
    """Normalized observation history, flattened from (..., hist, obs_dim) to (..., hist * obs_dim)."""
    return rn_apply(bufs["obs_norm"], obs_hist).reshape(obs_hist.shape[:-2] + (-1,))


def pi_encode_vel_n(params, bufs, cmd, hn):
    """Velocity encoder latent from a command (..., 3) and ``hn``, the output of obs_hist_n."""
    return mx.tanh(mlp_apply(params["enc_vel"], mx.concatenate([rn_apply(bufs["goal_norm_vel"], cmd), hn], axis=-1), elu))


def pi_encode_vel(params, bufs, cmd, obs_hist):
    """Velocity encoder latent from a command (..., 3) and the observation history (..., hist, obs_dim)."""
    return pi_encode_vel_n(params, bufs, cmd, obs_hist_n(bufs, obs_hist))


def pi_apply(params, bufs, obs_hist, goal, ref, z=None):
    """Action a = dec(z, observation history), with z = enc(goal) unless given.

    ``ref`` = (offset of q_ref(f+1) in goal, q0, alpha) adds the reference skip (q_ref(f+1) - q0) / alpha
    (--pi_ref_skip). With ``ref`` = None, ``goal`` is only used to compute z."""
    o = obs_hist_n(bufs, obs_hist)
    if z is None:
        z = pi_encode(params, bufs, goal)
    a = mlp_apply(params["dec"], mx.concatenate([z, o], axis=-1), elu)
    if ref is None:
        return a
    off, q0, alpha = ref
    J = a.shape[-1]
    return a + (goal[..., off : off + J] - q0) / alpha


# ---------------------------------------------------------------- critic
def critic_init(rng, in_dim: int, vel_dim: int = 0):
    """Motion critic ``net``. With ``vel_dim`` > 0, also the velocity critic ``vel`` on inputs of that size."""
    params, bufs = {"net": mlp_init(rng, [in_dim, 256, 128, 1])}, {"norm": rn_init(in_dim)}
    if vel_dim:
        params["vel"] = mlp_init(rng, [vel_dim, 256, 128, 1])
        bufs["norm_vel"] = rn_init(vel_dim)
    return params, bufs


def critic_apply(params, bufs, x):
    return mlp_apply(params["net"], rn_apply(bufs["norm"], x), elu)[..., 0]


def critic_vel_apply(params, bufs, x):
    return mlp_apply(params["vel"], rn_apply(bufs["norm_vel"], x), elu)[..., 0]
