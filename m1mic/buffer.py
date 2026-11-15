"""Fixed-size replay buffer in numpy, laid out (time, world).

Kept in numpy so rows can be written in place (an MLX array would be copied on every write).
"""

from __future__ import annotations

import numpy as np

FIELDS = ("s", "s1", "a", "a_prev", "xi", "f", "ep_step", "cut")
CMD_FIELDS = ("cmd", "cmd_ref")  # velocity task: the step's command, and whether it came from the clip
START_FIELDS = ("s", "a_prev", "xi", "f")


def fields(buf):
    """Per-transition fields of ``buf``: FIELDS, plus CMD_FIELDS if present."""
    return FIELDS + (CMD_FIELDS if "cmd" in buf else ())


def init(T: int, N: int, sdim: int, J: int, xi_dim: int, cmd: bool = False):
    """Empty buffer of T steps for N worlds. ``cmd`` adds the velocity command fields."""
    z = lambda *shape, dt=np.float32: np.zeros((T, N) + shape, dt)  # noqa: E731
    buf = {"s": z(sdim), "s1": z(sdim), "a": z(J), "a_prev": z(J), "xi": z(xi_dim), "f": z(dt=np.int32),
           "ep_step": z(dt=np.int32), "cut": z(dt=bool), "ptr": 0, "size": 0}  # fmt: skip
    if cmd:
        buf["cmd"], buf["cmd_ref"] = z(3), z(dt=bool)
    return buf


def add(buf, tr):
    """Write one step of transitions (all worlds) at the write head."""
    T = buf["s"].shape[0]
    ptr = buf["ptr"]
    for k in fields(buf):
        buf[k][ptr] = tr[k]
    buf["ptr"], buf["size"] = (ptr + 1) % T, min(buf["size"] + 1, T)


def _row(buf, age):
    """Buffer row of the transition ``age`` steps after the oldest one."""
    T = buf["s"].shape[0]
    return (buf["ptr"] - buf["size"] + age) % T


def sample_windows(buf, rng: np.random.Generator, B: int, k: int):
    """B windows of k consecutive steps, shape (B, k, ...), for the world-model loss.

    ``mask`` is 1 on step j while no episode cut happened in steps 0..j-1.
    """
    N = buf["s"].shape[1]
    t0 = rng.integers(0, buf["size"] - k + 1, B)
    env = rng.integers(0, N, B)
    rows = _row(buf, t0[:, None] + np.arange(k))
    e = env[:, None]
    out = {name: buf[name][rows, e] for name in ("s", "s1", "a", "a_prev", "xi", "cut")}
    d = out["cut"].astype(np.float32)
    out["mask"] = np.cumprod(np.concatenate((np.ones((B, 1), np.float32), 1 - d[:, :-1]), -1), -1)
    return out


def sample_starts(buf, rng: np.random.Generator, B: int, hist: int, envs=None):
    """B imagination starts and the rows of their observation history, oldest first.

    The history stays inside the start's episode and the buffer; where it would run past either, the earliest
    available step is repeated. ``envs`` = (lo, hi) restricts the worlds to range(lo, hi).
    """
    lo, hi = (0, buf["s"].shape[1]) if envs is None else envs
    age = rng.integers(0, buf["size"], B)
    env = rng.integers(lo, hi, B)
    row = _row(buf, age)
    ep_step = buf["ep_step"][row, env]
    back = np.arange(hist - 1, -1, -1)
    back = np.minimum(back[None, :], ep_step[:, None])
    back = np.minimum(back, age[:, None])
    hrows = _row(buf, age[:, None] - back)
    henv = env[:, None]
    start = {k: buf[k][row, env] for k in START_FIELDS + (CMD_FIELDS if "cmd" in buf else ())}
    hist_d = {k: buf[k][hrows, henv] for k in START_FIELDS}
    return start, hist_d
