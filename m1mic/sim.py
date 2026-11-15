"""Batched G1 tracking simulator on the CPU: MuJoCo's C engine in a thread pool.

Physics state and episode bookkeeping are numpy arrays with
one row per world. Observations, rewards, terminations and reset poses are computed in MLX.
"""

from __future__ import annotations

import copy
import os
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mujoco
import numpy as np

from .mdp import MAX_STEP_REWARD, MAX_VEL_STEP_REWARD, unflatten_state
from .motion import BIN_FRAMES
from .rot import mat_to_quat, mm3, mv3, so3_exp, tr3

DECIMATION = 4
SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS  # (time, qpos, qvel, act)


class XiLayout:
    """Layout of the sim context xi = (friction 1, joint offsets J, CoM offset 3), normalized by center and half-range.

    Friction and CoM offset are drawn once per world, joint encoder offsets once per episode (BlinkMimic's G1 default).
    The world model and the critic see xi."""

    def __init__(self, meta: dict):
        J = len(meta["joint_names"])
        lo, hi = meta["dr"]["friction_range"]
        self.J = J
        self.dim = 1 + J + 3
        self.friction, self.joint_offset, self.com = slice(0, 1), slice(1, 1 + J), slice(1 + J, 4 + J)
        self.center_np = np.asarray([(lo + hi) / 2] + [0.0] * J + [0.0] * 3, np.float32)
        half = meta["dr"]["joint_offset"].get(".*", 0.01)
        self.offset_half = np.full((J,), max(half, 1e-6), np.float32)
        self.scale_np = np.asarray([(hi - lo) / 2] + [max(half, 1e-6)] * J + [0.05] * 3, np.float32)
        self.center, self.scale = mx.array(self.center_np), mx.array(self.scale_np)

    def normalize(self, xi):
        return (xi - self.center) / self.scale


class Physics:
    """Thread pool that steps batches of worlds. mj_step releases the GIL, so the threads run in parallel."""

    def __init__(self, mj_model, meta: dict, threads: int = 0):
        self.threads = threads if threads > 0 else (os.cpu_count() or 4)
        self.m = mj_model
        # one model copy per thread: each thread writes per-world friction and CoM offset into its own model
        self.models = [copy.copy(mj_model) for _ in range(self.threads)]
        self.datas = [mujoco.MjData(m) for m in self.models]
        self.pool = ThreadPoolExecutor(self.threads)
        self.nstate = mujoco.mj_stateSize(mj_model, SPEC)
        self.geoms = np.asarray(meta["robot_geom_ids"])
        g0, g1 = int(self.geoms.min()), int(self.geoms.max()) + 1
        self.geom_slice = slice(g0, g1) if g1 - g0 == len(self.geoms) else None  # contiguous ids: index with a slice
        self.body = int(meta["com_body_id"])
        self.ipos0 = mj_model.body_ipos[self.body].copy()
        self.act = np.asarray(meta["actuator_of_joint"])

    def step(self, phys, warm, target, fric, com, idx=None, nstep: int = DECIMATION):
        """Advance the worlds ``idx`` (default: all) by ``nstep`` physics steps, in place.

        ``phys`` (N, nstate) and ``warm`` (N, nv) hold every world's state and solver warm start. ``target`` holds the
        PD targets, rows aligned with ``idx``. ``fric`` (N,) and ``com`` (N, 3) are the per-world randomization."""
        idx = np.arange(phys.shape[0]) if idx is None else np.asarray(idx)
        if idx.size == 0:
            return
        target = np.asarray(target, np.float64)
        parts = np.array_split(np.arange(idx.size), min(self.threads, idx.size))
        geoms, gs, body, ipos0, act = self.geoms, self.geom_slice, self.body, self.ipos0, self.act
        mj_step, set_state, get_state = mujoco.mj_step, mujoco.mj_setState, mujoco.mj_getState

        def work(k):
            m, d = self.models[k], self.datas[k]
            gf, bi, ctrl, ws = m.geom_friction, m.body_ipos, d.ctrl, d.qacc_warmstart
            for j in parts[k]:
                i = idx[j]
                if gs is not None:
                    gf[gs, 0] = fric[i]
                else:
                    gf[geoms, 0] = fric[i]
                bi[body] = ipos0 + com[i]
                set_state(m, d, phys[i], SPEC)
                ws[:] = warm[i]  # warm start is not part of SPEC, carry it across steps ourselves
                ctrl[act] = target[j]
                mj_step(m, d, nstep)
                get_state(m, d, phys[i], SPEC)
                warm[i] = ws

        list(self.pool.map(work, range(len(parts))))


class Sim:
    """``num_envs`` G1 worlds stepped together at 50 Hz (4 physics substeps of 5 ms per control step).

    Training mode samples reset frames from the reset bins and randomizes. ``eval_mode`` resets exactly onto the
    reference, with no randomization and no timeouts. The env state ``st`` is a dict of numpy arrays, updated in
    place."""

    def __init__(self, physics: Physics, meta: dict, num_envs: int, mdp, eval_mode: bool = False, gamma: float = 0.99,
                 max_episode_steps: int = 500, fail_reset: float = 0.0, seed: int = 0):  # fmt: skip
        self.physics, self.meta, self.N, self.mdp = physics, meta, num_envs, mdp
        self.eval_mode = eval_mode
        self.no_ref = None  # (num_envs,) bool: velocity-task worlds, set by the trainer
        self.gamma, self.fail_reset = gamma, fail_reset
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.J = len(meta["joint_names"])
        self.layout = XiLayout(meta)
        m = physics.m
        self.nq, self.nv = m.nq, m.nv
        self.qadr = np.asarray(meta["qpos_adr"])
        self.dadr = np.asarray(meta["dof_adr"])
        self.fq, self.fd = meta["free_qpos_adr"], meta["free_dof_adr"]
        self.default_q = np.asarray(meta["default_joint_pos"], np.float32)
        self.alpha = np.asarray(meta["action_scale"], np.float32)
        self.com_range = np.asarray(meta["com_range"], np.float32)
        dr = meta["dr"]
        self.reset_pos, self.reset_rot = mx.array(np.asarray(dr["reset_pos"], np.float32)), mx.array(np.asarray(dr["reset_rot"], np.float32))
        self.reset_lin_vel = mx.array(np.asarray(dr["reset_lin_vel"], np.float32))
        self.reset_ang_vel = mx.array(np.asarray(dr["reset_ang_vel"], np.float32))
        self.reset_joint_pos, self.reset_joint_vel = dr["reset_joint_pos"], dr["reset_joint_vel"]
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ setup
    def init(self, motion):
        """Initial env state: draw per-world friction and CoM offset, reset every world onto the reference."""
        N, L = self.N, self.layout
        xi = np.zeros((N, L.dim), np.float32)
        if self.eval_mode:
            fric = np.ones((N, 1), np.float32)
            com = np.zeros((N, 3), np.float32)
        else:
            # floor friction stays 1.0 and MuJoCo takes the larger of a contact pair, so draws below 1.0 do nothing
            lo, hi = self.meta["dr"]["friction_range"]
            fric = self.rng.uniform(lo, hi, (N, 1)).astype(np.float32)
            com = self.rng.uniform(self.com_range[:, 0], self.com_range[:, 1], (N, 3)).astype(np.float32)
        xi[:, L.friction], xi[:, L.com] = fric, com
        nb = motion.num_bins
        d = mujoco.MjData(self.physics.m)
        phys0 = np.zeros(self.physics.nstate)
        mujoco.mj_getState(self.physics.m, d, phys0, SPEC)
        st = {
            "phys": np.tile(phys0, (N, 1)), "warm": np.zeros((N, self.nv)), "xi": xi,
            "f": np.zeros(N, np.int32), "ep_step": np.zeros(N, np.int32), "a_prev": np.zeros((N, self.J), np.float32),
            "start_bin": np.zeros(N, np.int32), "ep_disc_ret": np.zeros(N, np.float32), "ep_ret": np.zeros(N, np.float32),
            "bin_G": np.zeros(nb, np.float32), "bin_w_avg": np.full(nb, 1.0 / nb, np.float32), "bin_fail": np.zeros(nb, np.float32),
            "focus_q": np.full(nb, 1.0 / nb, np.float32), "focus_share": 0.0,
            "done_eval": np.zeros(N, bool), "failed_eval": np.zeros(N, bool),
        }  # fmt: skip
        # previous bin of the same clip (or the bin itself at a clip start), for failure-weighted resets
        prev = np.arange(nb) - 1
        bin_clip = motion.bin_clip_np
        self.bin_prev = np.where((prev >= 0) & (bin_clip[np.maximum(prev, 0)] == bin_clip), prev, prev + 1)
        self.reset(motion, st, np.arange(N))
        return st

    # ------------------------------------------------------------------ state
    def flat_state(self, st, idx=None):
        """World states as float32 rows (n, 13 + 2 J): p, quat (wxyz, w >= 0), v, world-frame w, q, qd."""
        ph = st["phys"] if idx is None else st["phys"][idx]
        qpos, qvel = ph[:, 1 : 1 + self.nq], ph[:, 1 + self.nq : 1 + self.nq + self.nv]
        quat = qpos[:, self.fq + 3 : self.fq + 7]
        quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
        quat = quat * np.where(quat[:, :1] < 0, -1.0, 1.0)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        R = np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(-1, 3, 3)  # fmt: skip
        # MuJoCo's free-joint angular velocity is in the body frame
        w_world = np.einsum("nij,nj->ni", R, qvel[:, self.fd + 3 : self.fd + 6])
        return np.concatenate((qpos[:, self.fq : self.fq + 3], quat, qvel[:, self.fd : self.fd + 3], w_world,
                               qpos[:, self.qadr], qvel[:, self.dadr]), -1).astype(np.float32)  # fmt: skip

    # ------------------------------------------------------------------ reset
    def sample_frames(self, motion, st, n: int):
        """Reset frames for n worlds: pick a bin, then a uniform frame inside it. Returns (frames, bins).

        Bins come from BlinkMimic's mixture of uniform, low-return softmax, its running average, recent failures and
        the probes' focus distribution."""
        nb = motion.num_bins
        G_max = MAX_STEP_REWARD / (1 - self.gamma)
        x = -25.0 * st["bin_G"].astype(np.float64) / G_max
        soft = np.exp(x - x.max())
        soft = soft / soft.sum()
        st["bin_w_avg"] = (st["bin_w_avg"] * 0.999 + 0.001 * soft).astype(np.float32)
        w_avg = st["bin_w_avg"].astype(np.float64)
        probs = 0.5 / nb + 0.4 * soft + 0.1 * w_avg / w_avg.sum()
        if self.fail_reset > 0:
            fail = np.bincount(self.bin_prev, weights=st["bin_fail"], minlength=nb)
            if fail.sum() > 0:
                probs = (1 - self.fail_reset) * probs + self.fail_reset * fail / max(fail.sum(), 1e-6)
        probs = (1 - st["focus_share"]) * probs + st["focus_share"] * st["focus_q"]
        cdf = np.cumsum(probs)
        u = self.rng.random(n) * cdf[-1]
        bins = np.minimum(np.searchsorted(cdf, u, side="right"), nb - 1).astype(np.int32)
        start = motion.bin_start_np[bins]
        end = np.minimum(start + BIN_FRAMES, motion.clip_end_np[motion.bin_clip_np[bins]])
        f = start + (self.rng.random(n) * (end - start)).astype(np.int32)
        return f.astype(np.int32), bins

    def reset(self, motion, st, idx, frames=None):
        """Reset the worlds ``idx`` onto reference frames, sampled unless ``frames`` is given."""
        idx = np.asarray(idx)
        n = idx.size
        if n == 0:
            return
        if frames is None:
            frames, bins = self.sample_frames(motion, st, n)
        else:
            frames = np.asarray(frames, np.int32)
            bins = motion.frame_bin_np[frames]
        p, R, v, w, q, qd = motion.ref_state(mx.array(frames))
        if not self.eval_mode:
            u = lambda half: (2 * mx.random.uniform(shape=(n, half.shape[0])) - 1) * half  # noqa: E731
            p = p + u(self.reset_pos)
            R = mm3(so3_exp(u(self.reset_rot)), R)
            v = v + u(self.reset_lin_vel)
            w = w + u(self.reset_ang_vel)
            q = q + (2 * mx.random.uniform(shape=(n, self.J)) - 1) * self.reset_joint_pos
            if self.reset_joint_vel > 0:
                qd = qd + (2 * mx.random.uniform(shape=(n, self.J)) - 1) * self.reset_joint_vel
        q = mx.clip(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        quat, w_local = mat_to_quat(R), mv3(tr3(R), w)
        mx.eval(p, quat, v, w_local, q, qd)
        ph = np.zeros((n, st["phys"].shape[1]))
        qpos, qvel = ph[:, 1 : 1 + self.nq], ph[:, 1 + self.nq : 1 + self.nq + self.nv]  # views, time stays 0
        qpos[:, self.fq : self.fq + 3], qpos[:, self.fq + 3 : self.fq + 7] = np.asarray(p), np.asarray(quat)
        qpos[:, self.qadr] = np.asarray(q)
        qvel[:, self.fd : self.fd + 3], qvel[:, self.fd + 3 : self.fd + 6] = np.asarray(v), np.asarray(w_local)
        qvel[:, self.dadr] = np.asarray(qd)
        st["phys"][idx] = ph
        st["warm"][idx] = 0.0
        if not self.eval_mode:
            L = self.layout
            st["xi"][idx, L.joint_offset] = ((2 * self.rng.random((n, self.J)) - 1) * L.offset_half).astype(np.float32)
        st["f"][idx], st["ep_step"][idx], st["a_prev"][idx], st["start_bin"][idx] = frames, 0, 0.0, bins
        st["ep_disc_ret"][idx], st["ep_ret"][idx] = 0.0, 0.0

    # ------------------------------------------------------------------ step
    def observe(self, motion, st, noise: bool = False, s=None):
        """Policy observation of every world as an MLX array, with observation noise if ``noise``.
        ``s``: flat states, if already computed."""
        s = self.flat_state(st) if s is None else s
        no_ref = None if self.no_ref is None else mx.array(self.no_ref)
        o = self.mdp.obs(motion, unflatten_state(mx.array(s)), mx.array(st["a_prev"]), mx.array(st["xi"]), mx.array(st["f"]), no_ref=no_ref)
        return self.mdp.add_noise(o) if noise else o

    def n_motion(self) -> int:
        """Number of motion-tracking worlds. They come first, the velocity-command worlds (``no_ref``) after them."""
        return self.N if self.no_ref is None else int((~np.asarray(self.no_ref)).sum())

    def label(self, motion, s, s1, f1, a, a_prev, xi, cmd=None):
        """Reward and termination (MLX) of the transitions into the flat states s1 at frames f1.

        ``cmd``: command of the velocity worlds (the last N - n_motion). They get the velocity reward and the fall
        termination instead of the tracking ones."""
        S1 = unflatten_state(s1)
        nm = self.n_motion()
        if cmd is None or nm == self.N:
            return self.mdp.reward(motion, S1, f1, a, a_prev), self.mdp.termination(motion, S1, f1), None
        if nm == 0:
            return self.mdp.vel_reward(S1, cmd, a, a_prev), self.mdp.fall(S1), None
        Sm, Sv = tuple(x[:nm] for x in S1), tuple(x[nm:] for x in S1)
        r = mx.concatenate([self.mdp.reward(motion, Sm, f1[:nm], a[:nm], a_prev[:nm]), self.mdp.vel_reward(Sv, cmd, a[nm:], a_prev[nm:])])
        return r, mx.concatenate([self.mdp.termination(motion, Sm, f1[:nm]), self.mdp.fall(Sv)]), None

    def step_core(self, motion, st, a, cmd=None, s=None, label=None, idx=None):
        """Apply the action (numpy (N, J)), run 4 physics substeps and label the transition. Does not reset.

        ``s``: flat states before the step, if already computed.
        ``label(s, s1, f1, a, a_prev, xi)`` -> (reward, termination, extra) replaces ``self.label``. ``extra`` is
        evaluated with them and returned as out["also"].
        ``idx``: step the physics of these worlds only. Every world is still labelled and counted."""
        s = self.flat_state(st) if s is None else s
        f, xi, a_prev, ep_step = st["f"].copy(), st["xi"].copy(), st["a_prev"].copy(), st["ep_step"].copy()
        a = np.asarray(a, np.float32)
        target = self.default_q + xi[:, self.layout.joint_offset] + self.alpha * a
        self.physics.step(st["phys"], st["warm"], target if idx is None else target[idx], xi[:, 0], xi[:, self.layout.com], idx=idx)
        s1 = self.flat_state(st)
        f1 = f + 1
        xs = (mx.array(s), mx.array(s1), mx.array(f1), mx.array(a), mx.array(a_prev), mx.array(xi))
        r, term, extra = label(*xs) if label is not None else self.label(motion, *xs, cmd=cmd)
        mx.eval(r, term) if extra is None else mx.eval(r, term, extra)
        r, term = np.asarray(r), np.asarray(term)
        bad = ~np.isfinite(s1).all(-1)  # physics blew up: end the episode
        if bad.any():
            r, term = np.where(bad, 0.0, r).astype(np.float32), term | bad
        timeout = (f1 >= motion.clip_end_np[motion.clip_id_np[f]]) | (ep_step + 1 >= self.max_episode_steps)
        done = term | timeout
        out = {"s": s, "a": a, "a_prev": a_prev, "f": f, "s1": s1, "r": r, "term": term, "done": done, "ep_step": ep_step,
               "xi": xi, "cut": done, "also": extra}  # fmt: skip
        st["f"], st["ep_step"], st["a_prev"] = f1, ep_step + 1, a
        st["ep_disc_ret"] = st["ep_disc_ret"] + (self.gamma ** ep_step.astype(np.float32) * r).astype(np.float32)
        st["ep_ret"] = st["ep_ret"] + r
        if self.eval_mode:
            st["failed_eval"] = st["failed_eval"] | (term & ~st["done_eval"])
            st["done_eval"] = st["done_eval"] | done
        return out

    def step_tail(self, motion, st, out):
        """Update the bin statistics from ended episodes, reset them, return episode stats (BlinkMimic --bin_fix)."""
        done, term, f = out["done"], out["term"], out["f"]
        dw = done.astype(np.float32)
        T = st["ep_step"].astype(np.float32)
        vstats = {}
        if self.no_ref is not None:  # velocity worlds get their own stats and do not touch the reset bins
            vr = np.asarray(self.no_ref)
            dv = dw * vr
            vstats = {"vel_n_done": dv.sum(), "vel_ep_len_sum": (T * dv).sum(), "vel_fall_sum": float((term & vr).sum()),
                      "vel_ret_norm_sum": (st["ep_ret"] / (np.maximum(T, 1) * MAX_VEL_STEP_REWARD) * dv).sum()}  # fmt: skip
            dw = dw * ~vr
            term = term & ~vr
        any_done = bool((dw > 0).any())
        # timeouts bootstrap the return with the episode's mean reward
        boot = np.where(term, 0.0, self.gamma**T * (st["ep_ret"] / T) / (1 - self.gamma))
        G = st["ep_disc_ret"] + boot
        st["bin_G"] = _bin_ema_w(st["bin_G"], st["start_bin"], G, dw, 0.8)
        if self.fail_reset > 0:
            fb = motion.frame_bin_np[np.minimum(f, motion.num_frames - 1)]
            bf = st["bin_fail"] * 0.999 if any_done else st["bin_fail"]
            st["bin_fail"] = _bin_ema_w(bf, fb, term.astype(np.float32), dw, 0.9)
        stats = {"n_done": dw.sum(), "ep_len_sum": (T * dw).sum(), "ep_ret_norm_sum": (st["ep_ret"] / (T * MAX_STEP_REWARD) * dw).sum(),
                 "term_sum": float(term.sum()), **vstats}  # fmt: skip
        self.reset(motion, st, np.nonzero(done)[0])
        return stats


def _bin_ema_w(stat, bins, x, wt, old: float):
    """Per-bin EMA of the weighted mean of x over the samples in each bin. Bins without samples keep their value."""
    nb = stat.shape[0]
    sums = np.bincount(bins, weights=x * wt, minlength=nb)
    cnt = np.bincount(bins, weights=wt, minlength=nb)
    return np.where(cnt > 0, old * stat + (1 - old) * sums / np.maximum(cnt, 1), stat).astype(np.float32)
