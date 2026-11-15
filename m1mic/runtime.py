"""Training loop, train log (CSV, console, TensorBoard) and checkpoints.

Checkpoints use XMimic's npz layout, so its evaluation and rollout tools read them too.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

import mlx.core as mx
import numpy as np


from . import nets
from .algorithm import XMimic
from .config import Config
from .tblog import TBWriter

# train_log.csv column -> TensorBoard tag
TB_TAGS = {
    "ep_ret": "Train/mean_reward", "ep_len": "Train/mean_episode_length", "term_frac": "Episode_Termination/terminated",
    "sim_r": "Episode_Reward/step_reward", "v": "Loss/value_function", "pi": "Loss/surrogate", "lr": "Loss/learning_rate",
    "pi_lr": "Loss/policy_learning_rate", "wm": "Loss/world_model", "wm8": "Loss/world_model_8step",
    "gn": "Loss/grad_norm", "ir": "Imagination/reward", "surv": "Imagination/survival", "nonfinite": "Train/nonfinite",
    "bc": "Loss/velocity_latent_bc", "v_vel": "Loss/velocity_value_function", "ir_vel": "Velocity/imagined_reward",
    "surv_vel": "Velocity/imagined_survival", "vel_r": "Velocity/step_reward", "vel_ep_len": "Velocity/mean_episode_length",
    "vel_ret": "Velocity/mean_reward", "vel_fall": "Velocity/fall_fraction", "it_per_s": "Perf/iterations_per_s",
    "sim_steps": "Perf/total_sim_steps", "wall_min": "Perf/wall_min",
}  # fmt: skip


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def export_policy(path, avg, meta_extra, test=None):
    """Save the averaged policy and its normalizers. ``test`` = (obs history, goal, action), a check for converters."""
    flat = {}
    if test is not None:
        flat["test_oh"], flat["test_g"], flat["test_a"] = (np.asarray(x, np.float32) for x in test)
    params, bufs = avg  # enc, dec and, with the velocity task, enc_vel; obs_norm, goal_norm [, goal_norm_vel]
    for name in params:
        for i, layer in enumerate(params[name]):
            flat[f"{name}.{i}.w"] = np.asarray(layer["w"], np.float32)
            flat[f"{name}.{i}.b"] = np.asarray(layer["b"], np.float32)
    for name in bufs:
        for k, v in bufs[name].items():
            flat[f"{name}.{k}"] = np.asarray(v, np.float32)
    np.savez(path + ".tmp.npz", **flat, meta=json.dumps(meta_extra))
    os.replace(path + ".tmp.npz", path)


def load_ckpt(path):
    """(policy parameters, normalizers, metadata) of a checkpoint, as MLX trees."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    names = sorted({k.split(".")[0] for k in z.files if k.count(".") == 2 and k.endswith(".w")})  # enc, dec [, enc_vel]
    params = {name: [{"w": mx.array(z[f"{name}.{i}.w"]), "b": mx.array(z[f"{name}.{i}.b"])}
                     for i in range(sum(1 for k in z.files if k.startswith(f"{name}.") and k.endswith(".w")))]
              for name in names}  # fmt: skip
    norms = sorted({k.split(".")[0] for k in z.files if k.endswith(".mean")})  # obs_norm, goal_norm [, goal_norm_vel]
    bufs = {n: {k: mx.array(z[f"{n}.{k}"]) for k in ("mean", "var", "count")} for n in norms}
    return params, bufs, meta


class RunLog:
    """Writes a log row every log interval, and the checkpoints."""

    VEL_COLS = ["bc", "v_vel", "ir_vel", "surv_vel", "vel_r", "vel_ep_len", "vel_ret", "vel_fall"]  # with --vel_share
    COLS = ["iteration", "wall_min", "sim_steps", "wm", "wm8", "pi", "v", "ir", "gn", "surv", "sim_r", "ep_len", "ep_ret", "term_frac",
            "it_per_s", "lr", "pi_lr", "nonfinite"]  # fmt: skip

    def __init__(self, xm: XMimic, cfg: Config, t0: float, args_meta: dict):
        self.xm, self.cfg, self.t0, self.args_meta = xm, cfg, t0, args_meta
        self.cols = self.COLS + (self.VEL_COLS if cfg.vel_share > 0 else [])
        self.f = open(os.path.join(cfg.log_dir, "train_log.csv"), "a")
        self.f.write(",".join(self.cols) + "\n")
        self.tb = TBWriter(cfg.log_dir)
        self.acc, self.t_log, self.it_log, self.decay_at, self.sim_log, self.wall_log = [], time.time(), 0, -1, 0.0, 0.0
        self.times_log = dict(xm.times)
        # fixed inputs for the checkpoint's test vector
        rng = np.random.default_rng(1234)
        oh = mx.array((0.5 * rng.standard_normal((4, cfg.hist, xm.obs_dim))).astype(np.float32))
        frames = np.minimum(np.asarray([100, 5000, 100000, 300000]), xm.motion.num_frames - 1).astype(np.int32)
        self.test_in = (oh, xm.motion.goal(mx.array(frames), cfg.goal_frames))

    def add(self, it: int, st: dict):
        """The statistics of iteration ``it`` (1-based count of finished iterations); returns the training minutes so far."""
        c = self.cfg
        now = time.time()
        self.acc.append(st)
        if it % c.log_every == 0 or it >= c.iterations:
            s = {k: np.asarray([a[k] for a in self.acc], np.float64) for k in st}
            self.acc = []
            ips = (it - self.it_log) / (now - self.t_log)
            self.t_log, self.it_log = now, it
            nd = float(s["n_done"].sum())
            ep_len = float(s["ep_len_sum"].sum()) / max(nd, 1)
            ep_ret = float(s["ep_ret_norm_sum"].sum()) / max(nd, 1) if nd > 0 else float("nan")
            row = {"iteration": it, "wall_min": (now - self.t0) / 60, "sim_steps": float(s["sim_steps"][-1]),
                   "wm": float(s["wm"].mean()), "wm8": float(s["wm8"].mean()), "pi": float(s["pi"].mean()), "v": float(s["v"].mean()),
                   "ir": float(s["ir"].mean()), "gn": float(s["gn"].mean()), "surv": float(s["surv"].mean()),
                   "sim_r": float(s["r"].sum() / max(s["n_coll"].sum(), 1)), "ep_len": ep_len, "ep_ret": ep_ret,
                   "term_frac": float(s["term_sum"].sum()) / max(nd, 1), "it_per_s": ips, "lr": float(s["lr"][-1]),
                   "pi_lr": float(s["pi_lr"][-1]), "nonfinite": int(s["nonfinite"][-1])}  # fmt: skip
            if "bc" in self.cols:  # the velocity task (vel_ret: normalized by the velocity reward's maximum)
                nan = float("nan")
                vd = float(s["vel_n_done"].sum()) if "vel_n_done" in s else 0.0
                for k in ("bc", "v_vel", "ir_vel", "surv_vel"):
                    row[k] = float(s[k].mean()) if k in s else nan
                row["vel_r"] = float(s["vel_r"].sum() / max(s["n_coll"].sum(), 1)) if "vel_r" in s else nan
                row["vel_ep_len"] = float(s["vel_ep_len_sum"].sum()) / vd if vd > 0 else nan
                row["vel_ret"] = float(s["vel_ret_norm_sum"].sum()) / vd if vd > 0 else nan
                row["vel_fall"] = float(s["vel_fall_sum"].sum()) / vd if vd > 0 else nan
            self.f.write(",".join(f"{row[k]:.6g}" if isinstance(row[k], float) else str(row[k]) for k in self.cols) + "\n")
            self.f.flush()
            tb = {f"{TB_TAGS.get(k, 'Extra/' + k)}": v for k, v in row.items() if k != "iteration"}
            dt = row["wall_min"] * 60 - self.wall_log
            if dt > 0:
                tb["Perf/total_fps"] = (row["sim_steps"] - self.sim_log) / dt  # world-steps per second
            self.sim_log, self.wall_log = row["sim_steps"], row["wall_min"] * 60
            if "ep_n_fail" in s:
                tb["Probe/failed_clips"] = float(s["ep_n_fail"][-1])
            self.tb.scalars(it, tb)
            n_it = max(len(s["wm"]), 1)
            ms = {k: (v - self.times_log[k]) / n_it * 1000 for k, v in self.xm.times.items()}
            self.times_log = dict(self.xm.times)
            probe = (f" | probe fails {int(s['ep_n_fail'][-1])}" if "ep_n_fail" in s else "") + (
                f" | vel ret {row['vel_ret']:.3f} len {row['vel_ep_len']:.1f} fall {row['vel_fall']:.2f} imag_r {row['ir_vel']:.3f} "
                f"bc {row['bc']:.4f}" if "bc" in row else "")
            print(f"it {it} | {row['wall_min']:.2f} min | wm {row['wm']:.4f} e8 {row['wm8']:.4f} | pi {row['pi']:.3f} v {row['v']:.4f} "
                  f"imag_r {row['ir']:.3f} surv {row['surv']:.3f} | sim_r {row['sim_r']:.3f} ep_len {ep_len:.1f} ep_ret {ep_ret:.3f} "
                  f"term {row['term_frac']:.2f} | {ips:.2f} it/s (ms/it: collect {ms['collect']:.0f} wm {ms['wm']:.0f} pi {ms['pi']:.0f} "
                  f"critic {ms['critic']:.0f} probe {ms['probe']:.0f}) | pi_lr {row['pi_lr']:.2e}{probe}", flush=True)  # fmt: skip
            da = int(s["decay_at"].max())
            if self.decay_at < 0 <= da:
                self.decay_at = da
                print(f"lr decay trigger at it {da} (ep_ret {ep_ret:.3f})", flush=True)
        if it % c.ckpt_every == 0 or it >= c.iterations:
            self.checkpoint(it, now - self.t0, float(st["sim_steps"]))
        return (now - self.t0) / 60

    def checkpoint(self, it, wall, sim_steps):
        avg = self.xm.avg
        test_a = nets.pi_apply(avg[0], avg[1], *self.test_in, self.xm.ref)
        mx.eval(test_a)
        export_policy(os.path.join(self.cfg.log_dir, f"ckpt_{it:06d}.npz"), avg,
                      {"iteration": it, "wall_time": wall, "sim_steps": sim_steps, "args": self.args_meta}, (*self.test_in, test_a))

    def close(self):
        self.f.close()
        self.tb.close()


def run(cfg: Config):
    os.makedirs(cfg.log_dir, exist_ok=True)
    meta = {**dataclasses.asdict(cfg), "backend": "mlx", "mlx_version": getattr(mx, "__version__", "")}
    write_json(os.path.join(cfg.log_dir, "args.json"), meta)
    t_proc = time.time()
    xm = XMimic(cfg)
    xm.init()
    print(f"setup {time.time() - t_proc:.1f} s | motion {xm.motion.num_frames} frames, {xm.motion.num_bins} bins, "
          f"{len(xm.motion.names)} clips | goal {xm.goal_dim} critic_in {xm.critic_in} wm_in {xm.wm_in} | {cfg.num_envs} worlds on "
          f"{xm.physics.threads} threads | {xm.P if xm.probes else 0} probe worlds | device {mx.default_device()}", flush=True)  # fmt: skip
    t0 = time.time()  # training clock; the first compiled updates include tracing
    xm.warmup(cfg.warmup_steps)
    print(f"warm-up: {cfg.warmup_steps} collection steps {time.time() - t0:.1f} s", flush=True)
    log = RunLog(xm, cfg, t0, meta)
    it = 0
    while it < cfg.iterations:
        st = xm.iteration()
        it += 1
        if log.add(it, st) >= cfg.max_minutes > 0 and it < cfg.iterations:
            print(f"time budget {cfg.max_minutes} min reached at it {it}", flush=True)
            if it % cfg.ckpt_every:
                log.checkpoint(it, time.time() - t0, xm.sim_steps)
            break
    log.close()
    print(f"done: {it} iterations in {(time.time() - t0) / 60:.2f} min", flush=True)
    return xm
