"""One training iteration: world-model learning and policy improvement through imagined rollouts.

Each iteration collects sim steps into the replay buffer, updates the world model, updates the policy by backprop
through imagined rollouts and fits the critic. Physics and buffer are numpy on the CPU, networks and updates MLX.
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from . import buffer as rb
from . import nets, optim
from .config import Config, floats, widths
from .kin import Kinematics
from .mdp import MDP, flatten_state, inv_dim, obs_dim, state_dim, unflatten_state
from .model import load_model
from .motion import CONTROL_DT, GOAL_Q_OFFSET, load_motion
from .nets import tmap
from .sim import Physics, Sim

# world-model error-scale floors: position, rotation, linear and angular velocity (joints are added in __init__)
ERR_FLOOR_PARTS = ((3, 3e-3), (3, 5e-3), (3, 2e-2), (3, 5e-2))
STAT_KEYS = ("r", "n_done", "ep_len_sum", "ep_ret_norm_sum", "term_sum")  # collection stats, summed per iteration
VEL_STAT_KEYS = ("vel_r", "vel_n_done", "vel_ep_len_sum", "vel_ret_norm_sum", "vel_fall_sum")  # velocity worlds


class Mirror:
    """Left-right mirror of states, joints and sim context for world-model augmentation (BlinkMimic's mirror.py)."""

    def __init__(self, joint_names, layout):
        names = list(joint_names)
        perm, sign = [], []
        for n in names:
            partner = n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
            perm.append(names.index(partner))
            sign.append(-1.0 if any(f"_{t}_" in n for t in ("roll", "yaw")) else 1.0)
        self.perm, self.sign = mx.array(np.asarray(perm, np.int32)), mx.array(np.asarray(sign, np.float32))
        self.M = mx.array(np.asarray([1.0, -1.0, 1.0], np.float32))
        self.J = layout.J

    def joints(self, x):
        return mx.take(x, self.perm, axis=-1) * self.sign

    def state(self, S):
        p, R, v, w, q, qd = S
        M = self.M
        return p * M, R * M[:, None] * M[None, :], v * M, -(w * M), self.joints(q), self.joints(qd)

    def xi(self, xi):
        J = self.J  # xi starts (friction, joint offsets, CoM offset): mirror the joints, negate CoM y
        return mx.concatenate([xi[..., :1], self.joints(xi[..., 1 : 1 + J]), xi[..., 1 + J : 2 + J], -xi[..., 2 + J : 3 + J],
                               xi[..., 3 + J :]], axis=-1)  # fmt: skip


class M1mic:
    """Model, motion library, simulators, training state and the steps of the training iteration."""

    def __init__(self, cfg: Config, probes: bool | None = None):
        """``probes``: build the probe simulator. Defaults to cfg.eprobe_share > 0."""
        self.cfg = c = cfg
        if c.cache_gb > 0:  # MLX caches freed Metal buffers, by default up to nearly all memory
            mx.set_cache_limit(int(c.cache_gb * 2**30))
        self.mj_model, self.meta = load_model()
        meta = self.meta
        self.J = J = len(meta["joint_names"])
        self.kin = Kinematics(meta)
        feet = [meta["tracked_bodies"].index(b) for b in ("left_ankle_roll_link", "right_ankle_roll_link")]
        self.physics = Physics(self.mj_model, meta, c.threads)
        self.sim = Sim(self.physics, meta, c.num_envs, None, gamma=c.gamma, fail_reset=c.fail_reset, seed=c.seed)
        self.layout = self.sim.layout
        self.mdp = MDP(self.kin, meta["default_joint_pos"], feet, self.layout.dim, torso=meta["tracked_bodies"].index("torso_link"))
        self.sim.mdp = self.mdp
        self.motion = load_motion(c.motion_dir, self.kin, feet, clips=c.clips)
        self.goal_dim = self.motion.goal_dim(c.goal_frames)
        self.obs_dim = obs_dim(J)
        self.critic_in = inv_dim(J) + self.mdp.critic_extra + self.goal_dim
        self.critic_vel_in = inv_dim(J) + nets.VEL_DIM + self.layout.dim  # state, command, sim context
        self.wm_in = inv_dim(J) + J + self.layout.dim
        self.wm_out = 12 + 2 * J
        self.ref = (GOAL_Q_OFFSET, mx.array(np.asarray(meta["default_joint_pos"], np.float32)),
                    mx.array(np.asarray(meta["action_scale"], np.float32)))  # fmt: skip
        if not c.pi_ref_skip:
            self.ref = None
        self.setup_vel()
        self.sim.no_ref = self.no_ref_rows(c.num_envs)
        self.mirror = Mirror(meta["joint_names"], self.layout)
        self.err_floor = mx.array(np.concatenate([np.full((n,), v, np.float32) for n, v in ERR_FLOOR_PARTS]
                                                 + [np.full((J,), 3e-3, np.float32), np.full((J,), 5e-2, np.float32)]))  # fmt: skip
        self.n_mirror = int(c.wm_batch * c.wm_mirror)
        self.probes = c.eprobe_share > 0 if probes is None else probes
        self.make_probe_tiles()
        if self.probes:
            self.penv = Sim(self.physics, meta, self.P, self.mdp, eval_mode=True, gamma=c.gamma, seed=c.seed + 2)
        self.rng = np.random.default_rng(c.seed + 1)
        # compiled functions are pure: their random numbers are drawn outside and passed in
        C = mx.compile if c.compile else (lambda f: f)
        self._act = C(self._act_fn)
        self._label = C(self._label_fn)
        self._wm_update = C(self._wm_update_fn)
        self._pi_update = C(self._pi_update_fn)
        self._critic_fit = C(self._critic_fit_fn)
        self._probe_act = C(self._probe_act_fn)
        self._probe_term = C(self._probe_term_fn)
        self.times = {k: 0.0 for k in ("collect", "wm", "pi", "critic", "probe")}

    def setup_vel(self):
        """Velocity task setup (SuperMimic's slot_mix).

        The last round(n * vel_share) rows of every batch use the velocity encoder, the rest the motion encoder.
        """
        c = self.cfg
        self.vel = c.vel_share > 0
        self.vel_lo = mx.array(np.asarray(floats(c.vel_lo), np.float32))
        self.vel_hi = mx.array(np.asarray(floats(c.vel_hi), np.float32))
        self.stat_keys = STAT_KEYS + (VEL_STAT_KEYS if self.vel else ())

    def n_vel(self, n: int) -> int:
        """Number of velocity rows in a batch of n (they come last)."""
        return int(round(n * self.cfg.vel_share)) if self.vel else 0

    def no_ref_rows(self, n: int):
        """(n,) bool, True on the velocity rows. None without the velocity task."""
        return np.arange(n) >= n - self.n_vel(n) if self.vel else None

    def vel_cmd(self, f):
        """Reference command of frames f (from the clip, in its heading frame), clipped to the command range."""
        return mx.clip(self.motion.vel_command(f, self.cfg.vel_window), self.vel_lo, self.vel_hi)

    def resample_cmd(self, env):
        """SuperMimic's command source (Appendix G.1), resampled at every reset and every vel_resample steps.

        With probability vel_ref_prob the clip's reference command, otherwise a held command uniform in the range
        (zero, i.e. standing, with probability vel_zero_prob).
        """
        c, rng = self.cfg, self.rng
        nv = env["cmd_ref"].shape[0]
        new = env["ep_step"][-nv:] % c.vel_resample == 0
        lo, hi = np.asarray(floats(c.vel_lo), np.float32), np.asarray(floats(c.vel_hi), np.float32)
        ref = rng.random(nv) < c.vel_ref_prob
        hold = (lo + rng.random((nv, 3)) * (hi - lo)).astype(np.float32)
        hold = np.where((rng.random(nv) < c.vel_zero_prob)[:, None], 0.0, hold).astype(np.float32)
        env["cmd_ref"] = np.where(new, ref, env["cmd_ref"])
        env["cmd_hold"] = np.where(new[:, None], hold, env["cmd_hold"])

    def make_probe_tiles(self):
        """Probe worlds, one per tile of ``eprobe_tile`` frames of every clip.

        ep_tile is each tile's first frame. ep_next is the next reset bin within the same clip, used to spread
        failures backwards.
        """
        c, mo = self.cfg, self.motion
        cs, ce = mo.clip_start_np, mo.clip_end_np
        self.ep_tile = np.concatenate([np.arange(cs[i], ce[i] - 1, c.eprobe_tile) for i in range(len(cs))]).astype(np.int32)
        # with more tiles than eprobe_max, each sweep probes a fresh random sample
        self.P = int(min(self.ep_tile.shape[0], c.eprobe_max))
        nb = mo.num_bins
        nxt = np.arange(nb) + 1
        bc = mo.bin_clip_np
        self.ep_next = np.where((nxt < nb) & (bc[np.minimum(nxt, nb - 1)] == bc), nxt, nxt - 1)

    # ------------------------------------------------------------------ state
    def init(self):
        """Create all training state as attributes."""
        c, J = self.cfg, self.J
        mx.random.seed(c.seed)
        rng = np.random.default_rng(c.seed)
        self.env = self.sim.init(self.motion)
        self.pi, self.pi_b = nets.pi_init(rng, self.obs_dim, self.goal_dim, J, c.hist, enc_hidden=widths(c.enc_hidden),
                                          dec_hidden=widths(c.dec_hidden), vel=self.vel, vel_hidden=widths(c.vel_enc_hidden))  # fmt: skip
        self.wm, self.wm_b = nets.wm_init(rng, self.wm_in, self.wm_out, widths(c.wm_hidden))
        self.critic, self.critic_b = nets.critic_init(rng, self.critic_in, self.critic_vel_in if self.vel else 0)
        self.pi_mask = {k: nets.muon_mask(v) for k, v in self.pi.items()}
        self.wm_mask = {"net": nets.muon_mask(self.wm["net"])}
        self.critic_mask = tmap(lambda _: False, self.critic)
        self.buf = rb.init(c.buffer_steps, c.num_envs, state_dim(J), J, self.layout.dim, cmd=self.vel)
        if self.vel:  # velocity command source, resampled on the first step
            nv = self.n_vel(c.num_envs)
            self.env["cmd_ref"], self.env["cmd_hold"] = np.zeros(nv, bool), np.zeros((nv, 3), np.float32)
        self.obs_hist = mx.zeros((c.num_envs, c.hist, self.obs_dim))
        self.avg, self.avg_n = (self.pi, self.pi_b), 0
        self.opt_pi, self.opt_wm = optim.init(self.pi, self.pi_mask), optim.init(self.wm, self.wm_mask)
        self.opt_v = optim.init(self.critic, self.critic_mask)
        self.it, self.step_acc, self.decay_at, self.dec_nd, self.dec_rs = 0, 0.0, -1, 0.0, 0.0
        self.nonfinite, self.sim_steps = 0, 0.0
        if self.probes:
            P = self.P
            self.ep = {"pst": self.penv.init(self.motion), "hist": mx.zeros((P, c.hist, self.obs_dim)),
                       "fail_f": np.full(P, -1, np.int32), "stop": np.zeros(P, np.int32), "fcap": np.zeros(P, np.int32), "left": 0,
                       "pending": False, "snap": self.avg, "ep_fail": np.zeros(self.motion.num_bins, np.float32), "n_fail": 0}  # fmt: skip
        mx.eval(self.pi, self.pi_b, self.wm, self.wm_b, self.critic, self.critic_b, self.opt_pi, self.opt_wm, self.opt_v)

    # ------------------------------------------------------------------ 1. collection
    def _act_fn(self, pi, pi_b, oh, s, a_prev, xi, f, fresh, n_obs, n_act, cmd_ref=None, cmd_hold=None):
        """Observation, history, goal, normalizer updates and the noisy action of every world."""
        c, mo = self.cfg, self.motion
        no_ref = None if self.sim.no_ref is None else mx.array(self.sim.no_ref)
        o = self.mdp.obs(mo, unflatten_state(s), a_prev, xi, f, no_ref=no_ref) + n_obs
        oh = mx.where(fresh[:, None, None], mx.broadcast_to(o[:, None], oh.shape), mx.concatenate([oh[:, 1:], o[:, None]], axis=1))
        nm = c.num_envs - self.n_vel(c.num_envs)
        g = mo.goal(f[:nm] if self.vel else f, c.goal_frames)
        pi_b = {**pi_b, "obs_norm": nets.rn_update(pi_b["obs_norm"], o), "goal_norm": nets.rn_update(pi_b["goal_norm"], g)}
        if not self.vel:
            a = nets.pi_apply(pi, pi_b, oh, g, self.ref)
            return mx.clip(a + n_act, -20, 20), oh, pi_b
        # motion rows first, then velocity rows (encoded from command and observation history)
        cmd = mx.where(cmd_ref[:, None], self.vel_cmd(f[nm:]), cmd_hold)
        pi_b["goal_norm_vel"] = nets.rn_update(pi_b["goal_norm_vel"], cmd)
        z = mx.concatenate([nets.pi_encode(pi, pi_b, g), nets.pi_encode_vel(pi, pi_b, cmd, oh[nm:])], axis=0)
        a = nets.pi_apply(pi, pi_b, oh, None, None, z=z)
        return mx.clip(a + n_act, -20, 20), oh, pi_b, cmd

    def _label_fn(self, wm_b, s, s1, f1, a, a_prev, xi, cmd=None):
        """Reward and termination of the simulated transitions, plus the world model's normalizer updates."""
        mo, mdp = self.motion, self.mdp
        S0, S1 = unflatten_state(s), unflatten_state(s1)
        if cmd is None:
            r = mdp.reward(mo, S1, f1, a, a_prev)
            term = mdp.termination(mo, S1, f1)
        else:
            nm = self.sim.n_motion()
            Sm, Sv = tuple(x[:nm] for x in S1), tuple(x[nm:] for x in S1)
            r = mx.concatenate([mdp.reward(mo, Sm, f1[:nm], a[:nm], a_prev[:nm]), mdp.vel_reward(Sv, cmd, a[nm:], a_prev[nm:])])
            term = mx.concatenate([mdp.termination(mo, Sm, f1[:nm]), mdp.fall(Sv)])
        wm_b = {"in_norm": nets.rn_update(wm_b["in_norm"], nets.wm_inputs(S0, a, self.layout.normalize(xi))),
                "out_norm": nets.rn_update(wm_b["out_norm"], nets.wm_targets(S0, S1, CONTROL_DT)),
                "inc_norm": nets.rn_update(wm_b["inc_norm"], nets.increments(S0, S1))}  # fmt: skip
        return r, term, wm_b

    def collect_step(self):
        """Step every world once with the current policy plus action noise.

        Transitions go to the replay buffer and update the normalizers. Ended episodes reset.
        """
        c, sim, env, N = self.cfg, self.sim, self.env, self.cfg.num_envs
        s = sim.flat_state(env)
        args = [self.pi, self.pi_b, self.obs_hist, mx.array(s), mx.array(env["a_prev"]), mx.array(env["xi"]), mx.array(env["f"]),
                mx.array(env["ep_step"] == 0), self.mdp.noise((N, self.obs_dim)), c.action_noise * mx.random.normal(shape=(N, self.J))]  # fmt: skip
        cmd = None
        if self.vel:
            self.resample_cmd(env)
            a, self.obs_hist, self.pi_b, cmd = self._act(*args, mx.array(env["cmd_ref"]), mx.array(env["cmd_hold"]))
            mx.eval(a, self.obs_hist, self.pi_b, cmd)
        else:
            a, self.obs_hist, self.pi_b = self._act(*args)
            mx.eval(a, self.obs_hist, self.pi_b)
        label = (lambda *x: self._label(self.wm_b, *x)) if cmd is None else (lambda *x: self._label(self.wm_b, *x, cmd))
        out = sim.step_core(self.motion, env, np.asarray(a), s=s, label=label)
        self.wm_b = out["also"]
        nm = N - self.n_vel(N)
        if self.vel:
            out["cmd"] = np.concatenate((np.zeros((nm, 3), np.float32), np.asarray(cmd)))
            out["cmd_ref"] = np.concatenate((np.zeros(nm, bool), env["cmd_ref"]))
        rb.add(self.buf, out)
        tail = sim.step_tail(self.motion, env, out)
        self.sim_steps += N
        if self.vel:
            return {"r": out["r"][:nm].mean() / 0.1, "vel_r": out["r"][nm:].mean() / 0.1, **tail}
        return {"r": out["r"].mean() / 0.1, **tail}

    def collect(self, n: int):
        """Run ``n`` collection steps and return the summed statistics."""
        acc = {k: 0.0 for k in self.stat_keys}
        for _ in range(n):
            s = self.collect_step()
            for k in acc:
                acc[k] += float(s[k])
        return acc

    # ------------------------------------------------------------------ 2. world model
    def wm_loss(self, wm, wm_b, w):
        """Open-loop prediction over each window. The error is in the true state's frame, normalized per dimension."""
        c = self.cfg
        S = unflatten_state(w["s"][:, 0])
        xi_n = self.layout.normalize(w["xi"])
        preds = []
        for j in range(c.wm_window):
            S = nets.wm_step(wm, wm_b, S, w["a"][:, j], xi_n[:, j], CONTROL_DT)
            preds.append(S)
        S_hat = tuple(mx.stack([P[i] for P in preds], axis=1) for i in range(6))
        std = mx.maximum(nets.rn_std(wm_b["inc_norm"], 1e-4), self.err_floor)
        e = ((nets.state_error(S_hat, unflatten_state(w["s1"])) / std) ** 2).mean(-1)
        return (e * w["mask"]).sum() / w["s"].shape[0], e[:, -1]

    def _wm_update_fn(self, wm, opt, wm_b, w, lr):
        c, mir, n = self.cfg, self.mirror, self.n_mirror
        if n > 0:
            w = dict(w)
            ms = lambda x: flatten_state(mir.state(unflatten_state(x)))  # noqa: E731
            for k_, fn in (("s", ms), ("s1", ms), ("a", mir.joints), ("xi", mir.xi)):
                w[k_] = mx.concatenate([fn(w[k_][:n]), w[k_][n:]], axis=0)
        (loss, e_last), grads = mx.value_and_grad(self.wm_loss)(wm, wm_b, w)
        grads, gn = optim.clip_by_global_norm(grads, 1.0)
        finite = mx.isfinite(gn)  # skip non-finite updates, as for the policy
        grads = tmap(lambda g: mx.where(finite, g, 0.0), grads)
        wm, opt = optim.step(wm, grads, opt, self.wm_mask, lr, betas=(0.9, 0.999), ok=finite)
        m = w["mask"][:, -1]
        return wm, opt, loss / c.wm_window, (e_last * m).sum() / mx.maximum(m.sum(), 1)

    def wm_updates(self, lr: float):
        """Run the iteration's ``wm_batches`` world-model updates. Returns the last one's statistics."""
        c = self.cfg
        lr = mx.array(lr, mx.float32)
        for _ in range(c.wm_batches):
            w = rb.sample_windows(self.buf, self.rng, c.wm_batch, c.wm_window)
            w = {k: mx.array(w[k]) for k in ("s", "s1", "a", "xi", "mask")}
            self.wm, self.opt_wm, loss, e8 = self._wm_update(self.wm, self.opt_wm, self.wm_b, w, lr)
            mx.eval(self.wm, self.opt_wm, loss, e8)
        return {"wm": loss.item(), "wm8": e8.item()}

    # ------------------------------------------------------------------ 3. policy and 4. critic
    def sample_valid_starts(self, B: int, envs=None):
        """B imagination starts from 2B candidates, preferring those whose ``horizon`` frames stay inside their clip.

        ``envs`` = (lo, hi) restricts the worlds to range(lo, hi).
        """
        mo = self.motion
        s, h = rb.sample_starts(self.buf, self.rng, 2 * B, self.cfg.hist, envs)
        ok = s["f"] + self.cfg.horizon <= mo.clip_end_np[mo.clip_id_np[s["f"]]]
        idx = np.argsort(~ok, kind="stable")[:B]
        return {k: v[idx] for k, v in s.items()}, {k: v[idx] for k, v in h.items()}

    def sample_policy_batch(self):
        """Imagination starts and the stored states of their observation histories.

        With the velocity task, motion rows start from motion worlds and velocity rows from velocity worlds.
        """
        B = self.cfg.pi_batch
        if not self.vel:
            start, h = self.sample_valid_starts(B)
        else:
            Bv, N = self.n_vel(B), self.cfg.num_envs
            nm = N - self.n_vel(N)
            parts = [self.sample_valid_starts(B - Bv, (0, nm)), self.sample_valid_starts(Bv, (nm, N))]
            start, h = ({k: np.concatenate([p[i][k] for p in parts]) for k in parts[0][i]} for i in range(2))
        return {k: mx.array(v) for k, v in start.items()}, {k: mx.array(v) for k, v in h.items()}

    def imagine(self, pi, pi_b, wm, wm_b, critic, critic_b, start, o_hist0, noise):
        """Roll the policy through the world model for ``horizon`` steps. Returns (loss, rollout data for the critic).

        The loss is minus the discounted imagined return, with soft terminations and a critic bootstrap at the end.

        With the velocity task the last rows are SuperMimic's RL velocity interface. Command and observation history
        feed the velocity encoder, and the decoder weights are stop-gradiented so the return trains only that encoder.
        Those rows use the velocity reward, the soft fall termination and the velocity critic.
        """
        c, mdp, mo, H, K = self.cfg, self.mdp, self.motion, self.cfg.horizon, self.cfg.goal_frames
        S0, xi, f0, a_prev0 = unflatten_state(start["s"]), start["xi"], start["f"], start["a_prev"]
        xi_n = self.layout.normalize(xi)
        last = mo.num_frames - 1  # clamp to the library; starts are valid, so this only guards the gather
        F = mx.minimum(f0[None] + mx.arange(H, dtype=f0.dtype)[:, None], last)  # reference frame of each imagined step (H, B)
        F1 = mx.minimum(F + 1, last)
        f_H = F1[-1]
        G = mo.goal(F, K)  # all rows: critic goal input (motion rows), latent-loss anchor (velocity rows)
        B = f0.shape[0]
        Bm = B - self.n_vel(B)  # motion rows first, then the velocity rows
        no_ref = self.no_ref_rows(B)
        no_ref = None if no_ref is None else mx.array(no_ref)
        Z = nets.pi_encode(pi, pi_b, G[:, :Bm])  # goal encodings of all steps in one batch
        sg = mx.stop_gradient
        if self.vel:
            # latent-loss anchor: motion encoder's latent of the velocity rows' own reference (no gradient),
            # counted only on rows whose command comes from the clip
            Z_anc = sg(nets.pi_encode(pi, pi_b, G[:, Bm:]))
            F_all = mx.concatenate([F, mx.minimum(F[-1:] + 1, last)], axis=0)
            ref_src, hold = start["cmd_ref"][Bm:], start["cmd"][Bm:]
            CMD_all = mx.where(ref_src[:, None], self.vel_cmd(F_all[:, Bm:]), hold)  # (H + 1, Bv, 3), no gradient
            ref_w = ref_src.astype(mx.float32)
            dec_sg = tmap(sg, pi["dec"])
        S, oh, a_prev = S0, o_hist0, a_prev0
        S_list, A_list, Ap_list, bc_sum = [], [], [], mx.zeros(())
        for t in range(H):
            if self.vel:
                hn = nets.obs_hist_n(pi_b, oh)
                z_v = nets.pi_encode_vel_n(pi, pi_b, CMD_all[t], hn[Bm:])
                bc_sum = bc_sum + (((z_v - Z_anc[t]) ** 2).mean(-1) * ref_w).sum()
                a = mx.concatenate([nets.mlp_apply(pi["dec"], mx.concatenate([Z[t], hn[:Bm]], axis=-1), nets.elu),
                                    nets.mlp_apply(dec_sg, mx.concatenate([z_v, hn[Bm:]], axis=-1), nets.elu)], axis=0)  # fmt: skip
            else:
                a = nets.pi_apply(pi, pi_b, oh, G[t], self.ref, z=Z[t])
            S1 = nets.wm_step(wm, wm_b, S, a, xi_n, CONTROL_DT)
            o1 = mdp.obs(mo, S1, a, xi, F1[t], no_ref=no_ref) + noise[t]
            oh = mx.concatenate([oh[:, 1:], o1[:, None]], axis=1)
            S_list.append(S1), A_list.append(a), Ap_list.append(a_prev)
            S, a_prev = S1, a
        S_H = S
        S_all = tuple(mx.stack([s_[i] for s_ in S_list], axis=0) for i in range(6))
        A, Ap = mx.stack(A_list, axis=0), mx.stack(Ap_list, axis=0)
        flat = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
        g_H = mo.goal(f_H, K)
        # one FK pass over steps 0..H: steps 1..H score the reward, all of them feed the critic
        S_c = tuple(mx.concatenate([S0[i][None], S_all[i]], axis=0) for i in range(6))
        fk_c = tuple(x.reshape((H + 1, B) + x.shape[1:]) for x in self.kin.forward(*(flat(x) for x in S_c)))
        bp_c = fk_c[0]
        fk = tuple(x[1:] for x in fk_c)
        if self.vel:
            rows = lambda x, lo, hi: flat(x[:, lo:hi])  # noqa: E731
            Sm, Sv = tuple(rows(x, 0, Bm) for x in S_all), tuple(rows(x, Bm, B) for x in S_all)
            fkm, fkv = tuple(rows(x, 0, Bm) for x in fk), tuple(rows(x, Bm, B) for x in fk)
            Hm = lambda x: x.reshape(H, -1)  # noqa: E731
            r = mx.concatenate([Hm(mdp.reward(mo, Sm, rows(F1, 0, Bm), rows(A, 0, Bm), rows(Ap, 0, Bm), w_contact=c.contact_reward, fk=fkm)),
                                Hm(mdp.vel_reward(Sv, flat(CMD_all[:H]), rows(A, Bm, B), rows(Ap, Bm, B), fk=fkv))], axis=1)  # fmt: skip
            d = mx.concatenate([Hm(mdp.soft_termination(mo, Sm, rows(F1, 0, Bm))), Hm(mdp.soft_fall(Sv))], axis=1)
            S_Hm, S_Hv = tuple(x[:Bm] for x in S_H), tuple(x[Bm:] for x in S_H)
            v_boot = mx.concatenate([
                nets.critic_apply(critic, critic_b, mx.concatenate([mdp.critic_features(mo, S_Hm, f_H[:Bm], xi_n[:Bm], bp=bp_c[H][:Bm]),
                                                                    g_H[:Bm]], axis=-1)),
                nets.critic_vel_apply(critic, critic_b, mdp.vel_critic_features(S_Hv, CMD_all[H], xi_n[Bm:]))])  # fmt: skip
        else:
            Sf = tuple(flat(x) for x in S_all)
            r = mdp.reward(mo, Sf, flat(F1), flat(A), flat(Ap), w_contact=c.contact_reward, fk=tuple(flat(x) for x in fk)).reshape(H, -1)
            d = mdp.soft_termination(mo, Sf, flat(F1)).reshape(H, -1)
            v_boot = nets.critic_apply(critic, critic_b, mx.concatenate([mdp.critic_features(mo, S_H, f_H, xi_n, bp=bp_c[H]), g_H], axis=-1))
        alive = mx.cumprod(1 - d, axis=0)
        c_t = mx.concatenate([mx.ones_like(alive[:1]), alive[:-1]], axis=0)
        disc = mx.power(c.gamma, mx.arange(H).astype(mx.float32))[:, None]
        Jr = (disc * c_t * r).sum(0) + c.gamma**H * alive[-1] * v_boot
        bc = c.vel_latent_bc * bc_sum / B  # latent loss summed over steps, zero on motion rows
        aux = {"S_c": S_c, "bp_c": bp_c, "F": F, "F1": F1, "G": G, "g_H": g_H, "xi_n": xi_n, "r": r, "d": d, "c_t": c_t,
               "surv": alive[-1], "bc": bc}  # fmt: skip
        if self.vel:
            aux["CMD_all"] = CMD_all
        return (-Jr.mean() + bc if self.vel else -Jr.mean()), aux

    def _pi_update_fn(self, pi, opt_pi, pi_b, wm, wm_b, critic, critic_b, start, hist, n_hist, noise, pi_lr):
        """BPTT policy update. Returns the new policy, optimizer state, statistics and the critic phase's data."""
        c, mdp, mo, H = self.cfg, self.mdp, self.motion, self.cfg.horizon
        B = start["f"].shape[0]
        Bm = B - self.n_vel(B)
        no_ref = self.no_ref_rows(B)
        no_ref = None if no_ref is None else mx.array(no_ref)[:, None]  # (B, hist)
        o_hist0 = mdp.obs(mo, unflatten_state(hist["s"]), hist["a_prev"], hist["xi"], hist["f"], no_ref=no_ref) + n_hist
        (loss, aux), grads = mx.value_and_grad(self.imagine)(pi, pi_b, wm, wm_b, critic, critic_b, start, o_hist0, noise)
        grads, gn = optim.clip_by_global_norm(grads, 1.0)
        if c.pi_clip_value > 0:
            grads = tmap(lambda g: mx.clip(g, -c.pi_clip_value, c.pi_clip_value), grads)
        finite = mx.isfinite(gn)  # skip non-finite updates (BlinkMimic --no_sync)
        grads = tmap(lambda g: mx.where(finite, g, 0.0), grads)
        pi, opt_pi = optim.step(pi, grads, opt_pi, self.pi_mask, pi_lr, betas=(c.pi_beta1, 0.95), ok=finite)
        # critic inputs: states, frames and goals of steps 0..H of the imagined rollouts (no gradient)
        S_c, bp_c = aux["S_c"], aux["bp_c"]
        f_c = mx.concatenate([aux["F"][:1], aux["F1"]], axis=0)
        g_c = mx.concatenate([aux["G"], aux["g_H"][None]], axis=0)
        xi_c = mx.broadcast_to(aux["xi_n"], (H + 1,) + aux["xi_n"].shape)
        rews, dones = aux["r"], aux["d"]
        stats = {"pi": loss, "ir": ((aux["c_t"] * rews)[:, :Bm].sum(0) / H / 0.1).mean(), "gn": gn, "surv": aux["surv"][:Bm].mean(),
                 "finite": finite}  # fmt: skip
        if self.vel:
            m = lambda x: x[:, :Bm]  # noqa: E731
            cd = {"X": mx.concatenate([mdp.critic_features(mo, tuple(m(x) for x in S_c), m(f_c), m(xi_c), bp=m(bp_c)), m(g_c)], axis=-1),
                  "r": rews[:, :Bm], "d": dones[:, :Bm],
                  "Xv": mdp.vel_critic_features(tuple(x[:, Bm:] for x in S_c), aux["CMD_all"], xi_c[:, Bm:]),
                  "rv": rews[:, Bm:], "dv": dones[:, Bm:]}  # fmt: skip
            stats.update(bc=aux["bc"], ir_vel=((aux["c_t"] * rews)[:, Bm:].sum(0) / H / 0.1).mean(), surv_vel=aux["surv"][Bm:].mean())
        else:
            cd = {"X": mx.concatenate([mdp.critic_features(mo, S_c, f_c, xi_c, bp=bp_c), g_c], axis=-1), "r": rews, "d": dones}
        return pi, opt_pi, stats, tmap(mx.stop_gradient, cd)

    def _critic_fit_fn(self, critic, opt_v, critic_b, cd, perms, lr):
        """Critic phase: normalizer update, GAE(lambda) targets, epochs x minibatches of regression.

        With the velocity task its critic is fitted alongside with its own normalizer, targets, minibatches and
        gradient clip, in one optimizer step over both critics. Returns the last step's losses.
        """
        c, H = self.cfg, self.cfg.horizon
        vel = "Xv" in cd

        def targets_of(V, rews, dones):
            R, tg = V[-1], [None] * H
            for t in range(H - 1, -1, -1):
                R = rews[t] + c.gamma * (1 - dones[t]) * ((1 - c.lam) * V[t + 1] + c.lam * R)
                tg[t] = R
            return mx.stack(tg, axis=0)

        def rn_update(st, x):  # nets.rn_update, but a batch with non-finite entries leaves the normalizer unchanged
            ok = mx.isfinite(x).all()
            return tmap(lambda new, old: mx.where(ok, new, old), nets.rn_update(st, mx.where(ok, x, 0.0)), st)

        X = cd["X"]
        critic_b = {**critic_b, "norm": rn_update(critic_b["norm"], X)}
        Xf = X[:-1].reshape(-1, X.shape[-1])
        Yf = mx.stop_gradient(targets_of(nets.critic_apply(critic, critic_b, X), cd["r"], cd["d"])).reshape(-1)
        loss_m = lambda p, idx: ((nets.critic_apply({"net": p}, critic_b, Xf[idx]) - Yf[idx]) ** 2).mean()  # noqa: E731
        if vel:
            Xv = cd["Xv"]
            critic_b["norm_vel"] = rn_update(critic_b["norm_vel"], Xv)
            Xvf = Xv[:-1].reshape(-1, Xv.shape[-1])
            Yvf = mx.stop_gradient(targets_of(nets.critic_vel_apply(critic, critic_b, Xv), cd["rv"], cd["dv"])).reshape(-1)
            loss_v = lambda p, idx: ((nets.critic_vel_apply({"vel": p}, critic_b, Xvf[idx]) - Yvf[idx]) ** 2).mean()  # noqa: E731
        lv = mx.zeros(())
        for j in range(perms["m"].shape[0]):
            lm, gm = mx.value_and_grad(loss_m)(critic["net"], perms["m"][j])
            gm, gn = optim.clip_by_global_norm(gm, 1.0)
            g = {"net": gm}
            if vel:
                lv, gv = mx.value_and_grad(loss_v)(critic["vel"], perms["v"][j])
                g["vel"], gnv = optim.clip_by_global_norm(gv, 1.0)
                gn = gn + gnv
            finite = mx.isfinite(gn)  # skip non-finite steps
            g = tmap(lambda x: mx.where(finite, x, 0.0), g)
            critic, opt_v = optim.step(critic, g, opt_v, self.critic_mask, lr, ok=finite)
        return critic, critic_b, opt_v, lm, lv

    def critic_perms(self, n: int):
        """(epochs * minibatches, n // minibatches) int32 indices into n samples, a fresh permutation per epoch."""
        c = self.cfg
        mb = n // c.critic_minibatches
        return mx.array(np.concatenate([self.rng.permutation(n)[: mb * c.critic_minibatches].reshape(c.critic_minibatches, mb)
                                        for _ in range(c.critic_epochs)]).astype(np.int32))  # fmt: skip

    def policy_update(self, lr: float, pi_lr: float):
        """BPTT policy update, policy EMA, then the critic phase."""
        c, H, B = self.cfg, self.cfg.horizon, self.cfg.pi_batch
        t0 = time.perf_counter()
        start, hist = self.sample_policy_batch()
        n_hist, noise = self.mdp.noise((B, c.hist, self.obs_dim)), self.mdp.noise((H, B, self.obs_dim))
        self.pi, self.opt_pi, stats, cd = self._pi_update(self.pi, self.opt_pi, self.pi_b, self.wm, self.wm_b, self.critic,
                                                           self.critic_b, start, hist, n_hist, noise, mx.array(pi_lr, mx.float32))  # fmt: skip
        mx.eval(self.pi, self.opt_pi, stats, cd)
        self.avg = optim.ema(self.avg, (self.pi, self.pi_b), c.pi_ema, self.avg_n == 0)
        self.avg_n += 1
        mx.eval(self.avg)
        t1 = time.perf_counter()
        Bm = B - self.n_vel(B)
        perms = {"m": self.critic_perms(H * Bm)}
        if self.vel:
            perms["v"] = self.critic_perms(H * (B - Bm))
        self.critic, self.critic_b, self.opt_v, v, v_vel = self._critic_fit(self.critic, self.opt_v, self.critic_b, cd, perms,
                                                                            mx.array(lr, mx.float32))  # fmt: skip
        mx.eval(self.critic, self.critic_b, self.opt_v, v, v_vel)
        self.times["pi"] += t1 - t0
        self.times["critic"] += time.perf_counter() - t1
        finite = bool(stats.pop("finite").item())
        self.nonfinite += int(not finite)
        out = {k: x.item() for k, x in stats.items()}
        out["v"] = v.item()
        if self.vel:
            out["v_vel"] = v_vel.item()
        return out

    # ------------------------------------------------------------------ learning-rate schedule
    def lr_at(self, it: int) -> float:
        """Cosine schedule from lr down to 3 % of it over lr_decay_iters."""
        c = self.cfg
        if it >= c.lr_decay_iters:
            return c.lr * 0.03
        return c.lr * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * it / c.lr_decay_iters)))

    def pi_mult(self, it: int) -> float:
        """Policy LR multiplier: pi_lr_mult until the decay trigger, then a linear ramp to 1 over lr_decay_len."""
        c = self.cfg
        if self.decay_at < 0:
            return c.pi_lr_mult
        return c.pi_lr_mult + (1.0 - c.pi_lr_mult) * min(1.0, (it - self.decay_at) / max(c.lr_decay_len, 1))

    def decay_tick(self, it: int, cstats):
        """BlinkMimic's LR-decay trigger, checked at the end of each log interval.

        Fires when the mean normalized return of the episodes finished in the interval reaches lr_decay_ret, or at
        iteration lr_decay_by. The new decay_at applies from the next iteration.
        """
        c = self.cfg
        self.dec_nd += cstats["n_done"]
        self.dec_rs += cstats["ep_ret_norm_sum"]
        it1 = it + 1
        if it1 % c.log_every:
            return
        hit = self.dec_nd > 0 and self.dec_rs / max(self.dec_nd, 1.0) >= c.lr_decay_ret
        if c.lr_decay_by > 0:
            hit = hit or it1 >= c.lr_decay_by
        if self.decay_at < 0 and hit and c.lr_decay_ret > 0:
            self.decay_at = it1
        self.dec_nd, self.dec_rs = 0.0, 0.0

    # ------------------------------------------------------------------ evaluation-guided resets (probe sweeps)
    def probe_tiles(self):
        """First frames of this sweep's probe tiles: all of them, or a random sample of P for large libraries."""
        if self.P == self.ep_tile.shape[0]:
            return self.ep_tile
        return self.ep_tile[self.rng.choice(self.ep_tile.shape[0], self.P, replace=False)]

    def eprobe_tick(self, it: int):
        """BlinkMimic's --eprobe_async for one iteration.

        Applies a finished sweep's failures to the reset focus, starts a sweep every eprobe_every iterations from
        eprobe_start, then advances the running sweep by up to eprobe_async steps.
        """
        self.eprobe_control(it)
        self.eprobe_steps()

    def eprobe_control(self, it: int):
        """Finish a sweep (failure counts become the training worlds' reset focus) and start the next one."""
        c, mo, penv, ep, env = self.cfg, self.motion, self.penv, self.ep, self.env
        nb = mo.num_bins
        if ep["pending"] and ep["left"] == 0:
            # finish: decayed first-failure counts per bin, spread over the eprobe_win bins before each failure
            ff = ep["fail_f"]
            ep_fail = ep["ep_fail"] * c.eprobe_decay
            ep_fail = ep_fail + np.bincount(mo.frame_bin_np[ff[ff >= 0]], minlength=nb).astype(np.float32)
            w, q = ep_fail, ep_fail
            for _ in range(c.eprobe_win):
                w = w[self.ep_next] * (self.ep_next != np.arange(nb))
                q = q + w
            if ep_fail.sum() > 0:
                env["focus_q"], env["focus_share"] = (q / max(q.sum(), 1e-12)).astype(np.float32), c.eprobe_share
            ep["ep_fail"], ep["pending"], ep["n_fail"] = ep_fail, False, int((ff >= 0).sum())
        if it >= c.eprobe_start and it % c.eprobe_every == 0 and ep["left"] == 0 and not ep["pending"]:
            # start: snapshot the averaged policy, reset each probe on the reference a random lead before its tile
            lead = int(self.rng.integers(c.eprobe_lead_min, c.eprobe_lead_max + 1))
            tiles = self.probe_tiles()
            cid = mo.clip_id_np[tiles]
            f0 = np.maximum(tiles - lead, mo.clip_start_np[cid])
            stop = np.minimum(tiles + c.eprobe_tile, mo.clip_end_np[cid])
            fcap = np.minimum(stop, mo.clip_end_np[cid] - 1)
            pst = ep["pst"]
            pst["done_eval"][:], pst["failed_eval"][:] = False, False
            penv.reset(mo, pst, np.arange(self.P), frames=f0)
            o = penv.observe(mo, pst)
            hist = mx.broadcast_to(o[:, None], ep["hist"].shape)
            mx.eval(hist)
            ep.update(hist=hist, fail_f=np.full(self.P, -1, np.int32), stop=stop.astype(np.int32), fcap=fcap.astype(np.int32),
                      left=lead + c.eprobe_tile, snap=self.avg)  # fmt: skip

    def _probe_act_fn(self, snap_p, snap_b, hist, s, a_prev, xi, f, g, z):
        o = self.mdp.obs(self.motion, unflatten_state(s), a_prev, xi, f)
        h = mx.concatenate([hist[:, 1:], o[:, None]], axis=1)
        return nets.pi_apply(snap_p, snap_b, h, g, self.ref, z=z), h

    def _probe_term_fn(self, s1, f1):
        return self.mdp.termination(self.motion, unflatten_state(s1), f1)

    def _probe_label(self, s, s1, f1, a, a_prev, xi):
        """Label for probe worlds. Only the termination is used, the reward is never read."""
        return mx.zeros(f1.shape), self._probe_term(s1, f1), None

    def eprobe_steps(self):
        """Advance the running sweep by up to eprobe_async steps, recording each probe's first failure.

        Probe frames advance by one per step up to their cap and probes never reset mid-sweep, so the goals of all
        steps are known up front and encoded in one batch.
        """
        c, mo, penv, ep = self.cfg, self.motion, self.penv, self.ep
        k = min(c.eprobe_async, ep["left"])
        if k == 0:
            return
        t0 = time.perf_counter()
        snap_p, snap_b = ep["snap"]
        pst = ep["pst"]
        F = np.minimum(pst["f"][None] + np.arange(k, dtype=np.int32)[:, None], ep["fcap"][None])  # (k, P)
        G = mo.goal(mx.array(F), c.goal_frames)
        Z = nets.pi_encode(snap_p, snap_b, G)
        mx.eval(G, Z)
        for j in range(k):
            s = penv.flat_state(pst)
            a, ep["hist"] = self._probe_act(snap_p, snap_b, ep["hist"], mx.array(s), mx.array(pst["a_prev"]), mx.array(pst["xi"]),
                                            mx.array(pst["f"]), G[j], Z[j])  # fmt: skip
            mx.eval(a, ep["hist"])
            a = np.asarray(a)
            alive, f_now = ~pst["done_eval"], pst["f"].copy()
            out = penv.step_core(mo, pst, a, s=s, label=self._probe_label)
            failed = alive & (out["term"] | ~np.isfinite(a).all(-1))
            ep["fail_f"] = np.where(failed, f_now, ep["fail_f"])
            pst["done_eval"] = pst["done_eval"] | (pst["f"] >= ep["stop"])
            pst["f"] = np.minimum(pst["f"], ep["fcap"])
        ep["left"] -= k
        ep["pending"] = ep["pending"] or ep["left"] == 0
        self.times["probe"] += time.perf_counter() - t0

    # ------------------------------------------------------------------ one iteration
    def iteration(self):
        """One training iteration: probes, collection, LR-decay trigger, world model, policy and critic."""
        c, it = self.cfg, self.it
        lr = float(self.lr_at(it))
        pi_lr = lr * self.pi_mult(it)
        acc = self.step_acc + 1.0 / c.utd  # BlinkMimic's schedule: floor(acc) steps now, remainder carries over
        n = int(np.floor(acc))
        self.step_acc = acc - n
        if self.probes:
            self.eprobe_tick(it)
        t0 = time.perf_counter()
        cstats = self.collect(n)
        t1 = time.perf_counter()
        self.decay_tick(it, cstats)
        wstats = self.wm_updates(lr)
        self.times["collect"] += t1 - t0
        self.times["wm"] += time.perf_counter() - t1
        pstats = self.policy_update(lr, pi_lr)
        stats = {**wstats, **pstats, "n_coll": float(n), **cstats, "lr": lr, "pi_lr": pi_lr, "decay_at": float(self.decay_at),
                 "sim_steps": self.sim_steps, "nonfinite": float(self.nonfinite)}  # fmt: skip
        if self.probes:
            stats["ep_n_fail"], stats["ep_left"] = float(self.ep["n_fail"]), float(self.ep["left"])
        self.it = it + 1
        return stats

    def warmup(self, n_steps: int):
        """Run ``n_steps`` collection steps with the initial policy before the first update."""
        return self.collect(n_steps)
