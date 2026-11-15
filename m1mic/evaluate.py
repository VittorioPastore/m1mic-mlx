"""Checkpoint evaluation on the CPU simulator with BlinkMimic's protocol.

    python -m m1mic.evaluate <run_dir> [--every 2] [--last 4] [--clips walk,run]
    python -m m1mic.evaluate <run_dir> --task velocity [--reps 4] [--grid 15]

Each clip is played once from its first frame with friction 1.0, no randomization and no noise. A clip fails if the
training termination triggers. Normalized return = reward sum / ((clip length - 1) * MAX_STEP_REWARD). Writes
<run_dir>/eval_mlx.csv. There is one world per checkpoint and clip, since MuJoCo's C engine is deterministic.

--task velocity runs SuperMimic's step test (Appendix G.2) on a run trained with --vel_share: from standing, hold a
fixed command for 5 s. Writes eval_mlx_vel_step.csv and vel_step_<iteration>.csv.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import mlx.core as mx
import numpy as np

from . import nets
from .config import floats
from .kin import Kinematics
from .mdp import MAX_STEP_REWARD, MDP, unflatten_state
from .model import load_model
from .motion import GOAL_Q_OFFSET, load_motion
from .rot import mv3, tr3, yaw_mat
from .runtime import load_ckpt
from .sim import Physics, Sim


def setup(cfg: dict, motion_dir=None, clips=None, threads: int = 0):
    """Build (physics, meta, kinematics, feet, motion, reference skip) from a run's args."""
    mx.set_cache_limit(2**30)
    mj, meta = load_model()
    kin = Kinematics(meta)
    feet = [meta["tracked_bodies"].index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
    motion = load_motion(motion_dir or cfg["motion_dir"], kin, feet, clips=cfg.get("clips", "") if clips is None else clips)
    ref = (GOAL_Q_OFFSET, mx.array(np.asarray(meta["default_joint_pos"], np.float32)), mx.array(np.asarray(meta["action_scale"], np.float32)))
    if not cfg.get("pi_ref_skip", 1):
        ref = None
    return Physics(mj, meta, threads), meta, kin, feet, motion, ref


def make_mdp(meta, kin, feet, xi_dim):
    return MDP(kin, meta["default_joint_pos"], feet, xi_dim, torso=meta["tracked_bodies"].index("torso_link"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
    ap.add_argument("--last", type=int, default=0, help="only the last n of them (0: all)")
    ap.add_argument("--motion_dir", default=None, help="default: the run's own")
    ap.add_argument("--clips", default=None, help="clip-name prefixes to evaluate on (default: the run's own clips)")
    ap.add_argument("--group", type=int, default=8, help="checkpoints per batch")
    ap.add_argument("--threads", type=int, default=0, help="physics threads (0: one per CPU core)")
    ap.add_argument("--task", default="motion", choices=("motion", "velocity"), help="the encoder to evaluate")
    ap.add_argument("--reps", type=int, default=4, help="--task velocity: trials per command")
    ap.add_argument("--grid", type=int, default=15, help="--task velocity: commands per axis of each plane")
    args = ap.parse_args(argv)
    files = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.npz")))[:: args.every]
    if args.last > 0:
        files = files[-args.last :]
    assert files, args.run_dir
    cfg = json.loads(str(np.load(files[0])["meta"]))["args"]
    physics, meta, kin, feet, motion, ref = setup(cfg, args.motion_dir, args.clips, args.threads)
    if args.task == "velocity":
        assert cfg.get("vel_share", 0) > 0 and "vel_lo" in cfg, "--task velocity needs a run trained with --vel_share (RL velocity encoder)"
        return step_test(args, files, cfg, physics, meta, kin, feet, motion)
    C, K, Hh = len(motion.names), cfg["goal_frames"], cfg["hist"]
    groups = [files[i : i + args.group] for i in range(0, len(files), args.group)]
    clip_len = motion.clip_end_np - motion.clip_start_np + 1
    rows, t_all = [], time.time()
    for grp in groups:
        cks = [load_ckpt(f) for f in grp]
        M = len(cks)
        N = M * C  # world n: checkpoint n // C, clip n % C
        sim = Sim(physics, meta, N, None, eval_mode=True)
        mdp = sim.mdp = make_mdp(meta, kin, feet, sim.layout.dim)
        clip_of = np.arange(N) % C
        st = sim.init(motion)
        sim.reset(motion, st, np.arange(N), frames=motion.clip_start_np[clip_of])
        clip_end = motion.clip_end_np[clip_of]
        params, bufs = [c[0] for c in cks], [c[1] for c in cks]

        def act(params, bufs, hist, s, a_prev, xi, f, first):
            o = mdp.obs(motion, unflatten_state(s), a_prev, xi, f)
            h = mx.where(first, mx.broadcast_to(o[:, None], hist.shape), mx.concatenate([hist[:, 1:], o[:, None]], axis=1))
            g = motion.goal(f, K)
            a = [nets.pi_apply(params[m], bufs[m], h[m * C : (m + 1) * C], g[m * C : (m + 1) * C], ref) for m in range(M)]
            return mx.concatenate(a, axis=0), h

        def label(s, s1, f1, a, a_prev, xi):
            S1 = unflatten_state(s1)
            return mdp.reward(motion, S1, f1, a, a_prev), mdp.termination(motion, S1, f1)

        act_c, label_c = mx.compile(act), mx.compile(label)
        hist = mx.zeros((N, Hh, mdp.obs_dim))
        ret = np.zeros(N)
        t0, n_steps, first = time.time(), 0, True
        while not st["done_eval"].all():
            s = sim.flat_state(st)
            a, hist = act_c(params, bufs, hist, mx.array(s), mx.array(st["a_prev"]), mx.array(st["xi"]), mx.array(st["f"]), mx.array(first))
            mx.eval(a, hist)
            alive = ~st["done_eval"]
            out = sim.step_core(motion, st, np.asarray(a), s=s, label=lambda *x: (*label_c(*x), None), idx=np.nonzero(alive)[0])
            st["f"] = np.minimum(st["f"], clip_end - 1)
            ret += np.where(alive, out["r"], 0.0)
            n_steps, first = n_steps + 1, False
        ret, failed = ret.reshape(M, C), st["failed_eval"].reshape(M, C)
        norm = ret / ((clip_len - 1) * MAX_STEP_REWARD)
        for m, (_, _, mt) in enumerate(cks):
            rows.append({"iteration": mt["iteration"], "train_wall_h": mt["wall_time"] / 3600, "norm_return": float(norm[m].mean()),
                         "norm_return_framew": float(ret[m].sum() / ((clip_len - 1).sum() * MAX_STEP_REWARD)),
                         "n_failed": float(failed[m].sum()), "failed": "|".join(motion.names[i] for i in np.nonzero(failed[m])[0]),
                         "per_clip": norm[m]})  # fmt: skip
            r = rows[-1]
            print(f"[eval] it {r['iteration']} ({r['train_wall_h'] * 60:.2f} min): return {r['norm_return']:.4f}, failed {int(r['n_failed'])}/{C}", flush=True)
        print(f"group of {M} checkpoints x {C} clips = {N} worlds: {n_steps} steps in {time.time() - t0:.0f} s", flush=True)
    rows.sort(key=lambda r: r["iteration"])
    out = os.path.join(args.run_dir, "eval_mlx.csv")
    with open(out, "w") as fh:
        fh.write("iteration,train_wall_h,norm_return,norm_return_framew,n_failed,failed\n")
        for r in rows:
            fh.write(f"{r['iteration']},{r['train_wall_h']:.6f},{r['norm_return']:.6f},{r['norm_return_framew']:.6f},{r['n_failed']},{r['failed']}\n")
    with open(os.path.join(args.run_dir, "eval_mlx_per_clip.csv"), "w") as fh:  # last checkpoint only
        r = rows[-1]
        fh.write("clip,frames,norm_return,failed\n")
        bad = set(r["failed"].split("|"))
        for i, name in enumerate(motion.names):
            fh.write(f"{name},{clip_len[i]},{r['per_clip'][i]:.6f},{int(name in bad)}\n")
    ok = [r["norm_return"] >= 0.82 for r in rows]
    strict = [r["norm_return"] >= 0.82 and r["n_failed"] == 0 for r in rows]
    first = next((r for r, o in zip(rows, ok) if o), None)
    held = next((rows[i] for i in range(len(rows) - 2) if all(strict[i : i + 3])), None)
    fmt = lambda r: "--" if r is None else f"{r['train_wall_h'] * 60:.2f} min (it {r['iteration']})"  # noqa: E731
    print(f"first >= 0.82: {fmt(first)} | >= 0.82 and 0 failed, held x3: {fmt(held)} | total eval {time.time() - t_all:.0f} s -> {out}")
    return rows


def step_grid(lo, hi, n: int):
    """Step-test commands (P, 3) and their plane index: an n x n grid over [lo, hi] on each pair of command axes,
    third axis zero, standing command dropped."""
    axes = [np.linspace(lo[i], hi[i], n, dtype=np.float32) for i in range(3)]
    cmds, plane = [], []
    for p, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        a, b = np.meshgrid(axes[i], axes[j], indexing="ij")
        c = np.zeros((n * n, 3), np.float32)
        c[:, i], c[:, j] = a.ravel(), b.ravel()
        keep = np.abs(c).sum(-1) > 0
        cmds.append(c[keep])
        plane += [p] * int(keep.sum())
    return np.concatenate(cmds), np.asarray(plane)


def standing_frame(motion, window: int):
    """Frame with the smallest reference command among those with the root at or above median height."""
    speed, CH = [], 65536
    for s in range(0, motion.num_frames, CH):
        c = motion.vel_command(mx.arange(s, min(s + CH, motion.num_frames), dtype=mx.int32), window)
        speed.append(np.asarray(mx.sqrt((c * c).sum(-1))))
    speed = np.concatenate(speed)
    h = np.asarray(motion.root_p[:, 2])
    speed[h < np.median(h)] = np.inf
    return int(np.argmin(speed))


def step_test(args, files, cfg, physics, meta, kin, feet, motion, seconds: float = 5.0, settle: float = 3.0):
    """SuperMimic's velocity step test on each checkpoint. ``seconds`` per trial, error averaged after ``settle``."""
    lo, hi = np.asarray(floats(cfg["vel_lo"]), np.float32), np.asarray(floats(cfg["vel_hi"]), np.float32)
    cmds, plane = step_grid(lo, hi, args.grid)
    P, R, Hh = len(cmds), args.reps, cfg["hist"]
    T, T0 = int(round(seconds * 50)), int(round(settle * 50))  # 50 Hz steps, error averaged over T0..T-1
    f0 = standing_frame(motion, cfg["vel_window"])
    print(f"step test: {P} commands x {R} trials from frame {f0} ({motion.names[int(motion.clip_id_np[f0])]})", flush=True)
    rows, t_all = [], time.time()
    for path in files:
        params, bufs, mt = load_ckpt(path)
        N = R * P  # world n: trial n // P, command n % P
        sim = Sim(physics, meta, N, None, eval_mode=False)  # training randomization, so trials differ
        mdp = sim.mdp = make_mdp(meta, kin, feet, sim.layout.dim)
        sim.no_ref = np.ones(N, bool)
        cmd_np = np.tile(cmds, (R, 1))
        cmd = mx.array(cmd_np)
        st = sim.init(motion)
        sim.reset(motion, st, np.arange(N), frames=np.full(N, f0, np.int32))
        no_ref = mx.array(sim.no_ref)

        def act(hist, s, a_prev, xi, f, first):
            o = mdp.obs(motion, unflatten_state(s), a_prev, xi, f, no_ref=no_ref)
            h = mx.where(first, mx.broadcast_to(o[:, None], hist.shape), mx.concatenate([hist[:, 1:], o[:, None]], axis=1))
            return nets.pi_apply(params, bufs, h, None, None, z=nets.pi_encode_vel(params, bufs, cmd, h)), h

        def label(s, s1, f1, a, a_prev, xi):
            S1 = unflatten_state(s1)
            e = mx.concatenate([mv3(tr3(yaw_mat(S1[1])), S1[2])[:, :2], S1[3][:, 2:3]], axis=-1) - cmd
            return mdp.vel_reward(S1, cmd, a, a_prev), mdp.fall(S1), e

        act_c, label_c = mx.compile(act), mx.compile(label)
        hist = mx.zeros((N, Hh, mdp.obs_dim))
        fell, err = np.zeros(N, bool), np.zeros((N, 3))
        t0 = time.time()
        for t in range(T):
            s = sim.flat_state(st)
            a, hist = act_c(hist, mx.array(s), mx.array(st["a_prev"]), mx.array(st["xi"]), mx.array(st["f"]), mx.array(t == 0))
            mx.eval(a, hist)
            out = sim.step_core(motion, st, np.asarray(a), s=s, label=label_c)
            st["f"][:] = f0  # no reference to follow, keep the frame fixed
            fell |= out["term"]
            if t >= T0:
                err += np.asarray(out["also"])
        fell_r = fell.reshape(R, P)
        rel = np.linalg.norm(err / (T - T0), axis=-1) / np.linalg.norm(cmd_np, axis=-1)
        rel = np.where(fell_r, np.nan, rel.reshape(R, P))
        ok = ~fell_r.all(0)
        per_cmd = np.full(P, np.nan)
        per_cmd[ok] = np.nanmedian(rel[:, ok], 0)
        with open(os.path.join(args.run_dir, f"vel_step_{mt['iteration']:06d}.csv"), "w") as fh:
            fh.write("plane,vx,vy,wz,rel_err,n_fell\n")
            for i in range(P):
                fh.write(f"{plane[i]},{cmds[i, 0]:.4f},{cmds[i, 1]:.4f},{cmds[i, 2]:.4f},{per_cmd[i]:.6f},{int(fell_r[:, i].sum())}\n")
        rows.append({"iteration": mt["iteration"], "train_wall_h": mt["wall_time"] / 3600, "median_rel_err": float(np.nanmedian(per_cmd)),
                     "mean_rel_err": float(np.nanmean(per_cmd)), "fall_rate": float(fell.mean()), "n_all_fell": int((~ok).sum())})  # fmt: skip
        r = rows[-1]
        print(f"[step] it {r['iteration']} ({r['train_wall_h'] * 60:.2f} min): median rel err {r['median_rel_err']:.3f}, mean "
              f"{r['mean_rel_err']:.3f}, fall rate {r['fall_rate']:.3f}, all trials fell on {r['n_all_fell']}/{P} "
              f"({N} worlds, {T} steps in {time.time() - t0:.0f} s)", flush=True)  # fmt: skip
    out = os.path.join(args.run_dir, "eval_mlx_vel_step.csv")
    with open(out, "w") as fh:
        fh.write("iteration,train_wall_h,median_rel_err,mean_rel_err,fall_rate,n_all_fell\n")
        for r in sorted(rows, key=lambda r: r["iteration"]):
            fh.write(f"{r['iteration']},{r['train_wall_h']:.6f},{r['median_rel_err']:.6f},{r['mean_rel_err']:.6f},{r['fall_rate']:.6f},{r['n_all_fell']}\n")
    print(f"total eval {time.time() - t_all:.0f} s -> {out}")
    return rows


def rollout(ckpt, clip, start_s, seconds, motion_dir=None, task="motion", cmd=None):
    """Play ``seconds`` of ``clip`` from ``start_s`` under the evaluation protocol (used by tools/rollout.py).

    Returns robot qpos (T+1, nq), reference qpos ref (T+1, nq), reference frames f (T+1,) and alive (T+1,).
    With task="velocity" the velocity encoder follows ``cmd``: a fixed (vx, vy, yaw rate), a schedule
    [(t_seconds, (vx, vy, yaw rate)), ...], or None for the clip's own command clamped to the run's range.
    Velocity rollouts return cmd and vel (T+1, 3) instead of ref: heading-frame planar velocity and world yaw rate.
    """
    params, bufs, meta_ck = load_ckpt(ckpt)
    a = meta_ck["args"]
    physics, meta, kin, feet, motion, ref = setup(a, motion_dir, "", threads=1)
    vel = task == "velocity"
    assert not vel or a.get("vel_share", 0) > 0, "task velocity needs a checkpoint trained with --vel_share"
    sim = Sim(physics, meta, 1, None, eval_mode=True)
    sim.mdp = make_mdp(meta, kin, feet, sim.layout.dim)
    if vel:
        sim.no_ref = np.ones(1, bool)
    c = motion.names.index(clip)
    cs, ce = int(motion.clip_start_np[c]), int(motion.clip_end_np[c])
    f = min(cs + int(round(start_s * 50)), ce - 1)
    st = sim.init(motion)
    sim.reset(motion, st, np.arange(1), frames=np.asarray([f], np.int32))
    nq = sim.nq
    sched = None
    if vel and cmd is not None:
        sched = list(cmd) if isinstance(cmd[0], (tuple, list)) else [(0.0, tuple(cmd))]
    lo, hi = (mx.array(np.asarray(floats(a[k]), np.float32)) for k in ("vel_lo", "vel_hi")) if vel else (None, None)

    def command(t, f):
        if sched is None:
            return mx.clip(motion.vel_command(mx.array(np.asarray([f], np.int32)), a["vel_window"]), lo, hi)
        return mx.array(np.asarray([[x for ts, x in sched if ts <= t / 50][-1] if t / 50 >= sched[0][0] else sched[0][1]], np.float32))

    def ref_qpos(f):
        fm = mx.array(np.asarray([f], np.int32))
        return np.concatenate([np.asarray(motion.root_p[fm])[0], np.asarray(motion.root_quat[fm])[0], np.asarray(motion.q[fm])[0]])

    def measured(s):
        S = unflatten_state(mx.array(s))
        return np.asarray(mx.concatenate([mv3(tr3(yaw_mat(S[1])), S[2])[:, :2], S[3][:, 2:3]], axis=-1))[0]

    qpos = lambda: st["phys"][0, 1 : 1 + nq].copy()  # noqa: E731
    out = {"qpos": [qpos()], "f": [f], "alive": [True]}
    if vel:
        out["cmd"], out["vel"] = [np.asarray(command(0, f))[0]], [measured(sim.flat_state(st))]
    else:
        out["ref"] = [ref_qpos(f)]
    hist, alive = None, True
    for t in range(int(round(seconds * 50))):
        s = sim.flat_state(st)
        o = sim.observe(motion, st, s=s)
        hist = mx.broadcast_to(o[:, None], (1, a["hist"], o.shape[-1])) if hist is None else mx.concatenate([hist[:, 1:], o[:, None]], axis=1)
        fm = mx.array(st["f"])
        if vel:
            c_t = command(t, int(st["f"][0]))
            act = nets.pi_apply(params, bufs, hist, None, None, z=nets.pi_encode_vel(params, bufs, c_t, hist))
        else:
            c_t = None
            act = nets.pi_apply(params, bufs, hist, motion.goal(fm, a["goal_frames"]), ref)
        mx.eval(act, hist)
        o_ = sim.step_core(motion, st, np.asarray(act), cmd=c_t, s=s)
        st["f"] = np.minimum(st["f"], ce - 1)
        alive = alive and not bool(o_["term"][0])
        out["qpos"].append(qpos()), out["f"].append(int(st["f"][0])), out["alive"].append(alive)
        if vel:
            out["cmd"].append(np.asarray(c_t)[0]), out["vel"].append(measured(o_["s1"]))
        else:
            out["ref"].append(ref_qpos(int(st["f"][0])))
    return {k: np.asarray(v) for k, v in out.items()}


if __name__ == "__main__":
    main()
