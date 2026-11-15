"""Observation, reward and termination of the tracking task. MLX port of XMimic's mdp.py.

The same functions label simulator transitions and imagined world-model transitions. A state is the tuple
S = (p, R, v, w, q, qd): root position, root rotation matrix, world-frame root velocities, joint positions, velocities.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .motion import CONTROL_DT
from .rot import mat_to_6d, mat_to_quat, mm3, mv3, quat_to_mat, rot_angle_sq, tr3, yaw_mat

CONTACT_Z = 0.045
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT
BODY_POS_STD, BODY_ORI_STD, BODY_LINVEL_STD, BODY_ANGVEL_STD = 0.3, 0.4, 1.0, 3.14
ROOT_POS_STD, ROOT_ORI_STD = 0.3, 0.4
W_ACTION_RATE, W_JOINT_LIMIT, SOFT_LIMIT_FACTOR = -0.1, -2.0, 0.9
TERM_HEIGHT, TERM_TILT = 0.25, 0.8
# uniform noise half-widths (SuperMimic Table A.1): q, qd, angular velocity, gravity, previous action, ref orientation
NOISE = (0.01, 0.5, 0.2, 0.05, 0.0, 0.05)
# velocity task reward and termination (SuperMimic Table A.6), every term times CONTROL_DT
VEL_LIN_VAR, VEL_YAW_VAR, W_VEL_LIN, W_VEL_YAW = 0.5**2, 0.5, 2.0, 2.0
W_VEL_RATE, W_VEL_LEAN, W_VEL_TORSO_RATE = -0.3, -4.0, -0.05
MAX_VEL_STEP_REWARD = (W_VEL_LIN + W_VEL_YAW) * CONTROL_DT
FALL_HEIGHT, FALL_TILT = 0.35, 0.7  # fall: root height below 0.35 m, or 1 - R_zz above 0.7 (about 73 deg of tilt)


def state_dim(J: int) -> int:
    return 13 + 2 * J


def obs_dim(J: int) -> int:
    return 3 * J + 12


def inv_dim(J: int) -> int:
    return 13 + 2 * J


def flatten_state(S):
    p, R, v, w, q, qd = S
    return mx.concatenate([p, mat_to_quat(R), v, w, q, qd], axis=-1)


def unflatten_state(x):
    J = (x.shape[-1] - 13) // 2
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13 : 13 + J], x[..., 13 + J :]


def invariant_features(S):
    """Heading- and position-invariant state (..., 13 + 2 J): height, tilt (6D), body-frame velocities, joints."""
    p, R, v, w, q, qd = S
    Rt = tr3(R)
    return mx.concatenate([p[..., 2:3], mat_to_6d(mm3(tr3(yaw_mat(R)), R)), mv3(Rt, v), mv3(Rt, w), q, qd], axis=-1)


class MDP:
    """Observation, reward, hard and soft termination, and the critic's privileged features.

    The reward is BeyondMimic's tracking terms plus InstantTrack's contact term."""

    def __init__(self, kin, default_q, feet: list[int], xi_dim: int, torso: int | None = None):
        """``torso``: index of the torso among the tracked bodies, used by the velocity reward."""
        self.kin, self.torso = kin, torso
        J = len(default_q)
        self.J, self.xi_dim = J, xi_dim
        self.default_q = mx.array(np.asarray(default_q, np.float32))
        mid, half = (kin.q_lo_np + kin.q_hi_np) / 2, (kin.q_hi_np - kin.q_lo_np) / 2
        self.soft_lo, self.soft_hi = mx.array(mid - half * SOFT_LIMIT_FACTOR), mx.array(mid + half * SOFT_LIMIT_FACTOR)
        self.feet = mx.array(np.asarray(feet, np.int32))
        self.obs_dim = obs_dim(J)
        self.critic_extra = J + len(kin.tracked_idx) * 3 + 3 + 6 + xi_dim
        n = NOISE
        self.obs_noise = mx.array(np.concatenate((np.full(J, n[0]), np.full(J, n[1]), np.full(3, n[2]), np.full(3, n[3]),
                                                  np.full(J, n[4]), np.full(6, n[5]))).astype(np.float32))  # fmt: skip
        self.ref_6d_none = mx.array(np.eye(3, dtype=np.float32)[:, :2].reshape(6))  # ref orientation with no reference

    def obs(self, motion, S, a_prev, xi, f, no_ref=None):
        """Noise-free observation: q - q0 - offset, qd, body angular velocity, projected gravity, a_prev, ref orientation (6D).

        ``no_ref`` (bool, per world) marks worlds driven by the velocity command. They have no reference, so their
        reference orientation is the identity, which is what a perfect tracker would observe."""
        p, R, v, w, q, qd = S
        Rt = tr3(R)
        dq0 = xi[..., 1 : 1 + self.J]
        rel = mat_to_6d(mm3(Rt, motion.root_R[f]))
        if no_ref is not None:
            rel = mx.where(no_ref[..., None], self.ref_6d_none, rel)
        return mx.concatenate([q - self.default_q - dq0, qd, mv3(Rt, w), -R[..., 2, :], a_prev, rel], axis=-1)

    def noise(self, shape):
        """Uniform observation noise, shape (..., obs_dim)."""
        return (2 * mx.random.uniform(shape=shape) - 1) * self.obs_noise

    def add_noise(self, o):
        return o + self.noise(o.shape)

    def _anchored_ref(self, motion, p, R, f):
        rp, rR = motion.root_p[f], motion.root_R[f]
        dR = yaw_mat(mm3(R, tr3(rR)))
        anchor = mx.concatenate([p[..., :2], rp[..., 2:3]], axis=-1)
        ref_bp = anchor[..., None, :] + mv3(dR[..., None, :, :], motion.body_p[f] - rp[..., None, :])
        return rp, rR, dR, ref_bp

    def reward(self, motion, S, f, a, a_prev, w_contact: float = 0.0, terms: bool = False, fk=None):
        """Per-step reward (times dt) of S against reference frame f. w_contact > 0 adds InstantTrack's contact term.
        ``fk``: forward kinematics of S, if already computed."""
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd) if fk is None else fk
        rbp, rbR, rbv, rbw = motion.bodies(f)
        rp, rR = motion.root_p[f], motion.root_R[f]
        dR = yaw_mat(mm3(R, tr3(rR)))
        anchor = mx.concatenate([p[..., :2], rp[..., 2:3]], axis=-1)
        ref_bp = anchor[..., None, :] + mv3(dR[..., None, :, :], rbp - rp[..., None, :])
        ref_bR = mm3(dR[..., None, :, :], rbR)
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((rbv - bv) ** 2).sum(-1).mean(-1)
        e_av = ((rbw - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_track = (mx.exp(-e_pos / BODY_POS_STD**2) + mx.exp(-e_ori / BODY_ORI_STD**2) + mx.exp(-e_lv / BODY_LINVEL_STD**2)
                   + mx.exp(-e_av / BODY_ANGVEL_STD**2) + 0.5 * mx.exp(-e_rp / ROOT_POS_STD**2) + 0.5 * mx.exp(-e_ro / ROOT_ORI_STD**2))  # fmt: skip
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (mx.maximum(q - self.soft_hi, 0) + mx.maximum(self.soft_lo - q, 0)).sum(-1)
        r = (r_track + W_ACTION_RATE * r_rate + W_JOINT_LIMIT * r_lim) * CONTROL_DT
        if w_contact > 0:
            fz = mx.take(bp, self.feet, axis=-2)[..., 2]
            c_sim = mx.sigmoid((CONTACT_Z - fz) / 0.005)
            c_ref = motion.ref_contact[f]
            slip = (mx.take(bv, self.feet, axis=-2)[..., :2] ** 2).sum(-1)
            r_contact = (1 - (c_sim - c_ref) ** 2) + c_ref * mx.exp(-slip / 0.3**2)
            r = r + w_contact * r_contact.mean(-1) * CONTROL_DT
        if terms:
            return r, {"e_body_pos": e_pos, "e_body_ori": e_ori, "e_root_pos": e_rp, "track": r_track * CONTROL_DT}
        return r

    def termination(self, motion, S, f):
        """True where the root height or tilt is too far from the reference."""
        p, R = S[0], S[1]
        e_z = mx.abs(motion.root_p[f][..., 2] - p[..., 2])
        e_tilt = mx.abs(motion.root_R[f][..., 2, 2] - R[..., 2, 2])
        return (e_z > TERM_HEIGHT) | (e_tilt > TERM_TILT)

    def soft_termination(self, motion, S, f, tz: float = 0.02, tt: float = 0.04):
        """Differentiable termination probability for imagined rollouts (sigmoids around the hard thresholds)."""
        p, R = S[0], S[1]
        e_z = mx.abs(motion.root_p[f][..., 2] - p[..., 2])
        e_tilt = mx.abs(motion.root_R[f][..., 2, 2] - R[..., 2, 2])
        return 1 - (1 - mx.sigmoid((e_z - TERM_HEIGHT) / tz)) * (1 - mx.sigmoid((e_tilt - TERM_TILT) / tt))

    def critic_features(self, motion, S, f, xi_n, bp=None):
        """Invariant state, q_ref - q, body and root errors in the heading frame, ref orientation (6D), sim context.
        ``bp``: tracked body positions of S, if already computed."""
        p, R, v, w, q, qd = S
        Ht = tr3(yaw_mat(R))
        bp = self.kin.forward(p, R, v, w, q, qd)[0] if bp is None else bp
        rp, rR, _, ref_bp = self._anchored_ref(motion, p, R, f)
        e_body = mv3(Ht[..., None, :, :], ref_bp - bp).reshape(bp.shape[:-2] + (-1,))
        e_root = mv3(Ht, rp - p)
        rel = mat_to_6d(mm3(tr3(R), rR))
        return mx.concatenate([invariant_features(S), motion.q[f] - q, e_body, e_root, rel, xi_n], axis=-1)

    # ------------------------------------------------------------------ velocity task (SuperMimic Table A.6)
    def vel_reward(self, S, cmd, a, a_prev, fk=None, terms: bool = False):
        """Velocity-tracking reward (times dt) of S under cmd = (vx, vy, yaw rate).

        Rewards planar root velocity in the heading frame and world yaw rate matching the command. Penalizes action
        rate, soft joint limits, sideways torso lean and torso roll and pitch rates.
        ``fk``: forward kinematics of S, if already computed."""
        p, R, v, w, q, qd = S
        _, bR, _, bw = self.kin.forward(p, R, v, w, q, qd) if fk is None else fk
        Ht = tr3(yaw_mat(R))
        e_lin = ((mv3(Ht, v)[..., :2] - cmd[..., :2]) ** 2).sum(-1)
        e_yaw = (w[..., 2] - cmd[..., 2]) ** 2
        z_t = mv3(Ht, bR[..., self.torso, :, 2])  # torso z axis in the heading frame
        lean = z_t[..., 1] ** 2 / mx.maximum(z_t[..., 1] ** 2 + z_t[..., 2] ** 2, 1e-6)  # sin^2 of the sideways lean
        w_t = mv3(Ht, bw[..., self.torso, :])
        r_task = W_VEL_LIN * mx.exp(-e_lin / VEL_LIN_VAR) + W_VEL_YAW * mx.exp(-e_yaw / VEL_YAW_VAR)
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (mx.maximum(q - self.soft_hi, 0) + mx.maximum(self.soft_lo - q, 0)).sum(-1)
        r = (r_task + W_VEL_RATE * r_rate + W_JOINT_LIMIT * r_lim + W_VEL_LEAN * lean
             + W_VEL_TORSO_RATE * (w_t[..., 0] ** 2 + w_t[..., 1] ** 2)) * CONTROL_DT  # fmt: skip
        if terms:
            return r, {"e_lin": e_lin, "e_yaw": e_yaw, "task": r_task * CONTROL_DT}
        return r

    def fall(self, S):
        """Velocity task termination: root below FALL_HEIGHT or tilted past FALL_TILT."""
        p, R = S[0], S[1]
        return (p[..., 2] < FALL_HEIGHT) | (1 - R[..., 2, 2] > FALL_TILT)

    def soft_fall(self, S, tz: float = 0.02, tt: float = 0.04):
        """Differentiable fall probability for imagined rollouts, same form as soft_termination."""
        p, R = S[0], S[1]
        return 1 - (1 - mx.sigmoid((FALL_HEIGHT - p[..., 2]) / tz)) * (1 - mx.sigmoid((1 - R[..., 2, 2] - FALL_TILT) / tt))

    @staticmethod
    def vel_critic_features(S, cmd, xi_n):
        """Velocity critic input: invariant state, command, normalized sim context (SuperMimic Table A.4)."""
        return mx.concatenate([invariant_features(S), cmd, xi_n], axis=-1)
