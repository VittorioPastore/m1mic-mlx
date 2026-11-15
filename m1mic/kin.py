"""Differentiable forward kinematics of the G1 in MLX. The tree comes from the model meta."""

import mlx.core as mx
import numpy as np

from .rot import axis_angle_to_mat, cross, mm3, mv3


class Kinematics:
    def __init__(self, meta: dict):
        self.parent = list(meta["kin_parent"])
        self.joint_idx = list(meta["kin_joint_idx"])
        self.axes = [tuple(float(x) for x in a) for a in meta["kin_axes"]]
        off_p = np.asarray(meta["kin_off_p"], np.float32)
        off_R = np.asarray(meta["kin_off_R"], np.float32)
        self.off_p = [mx.array(p) for p in off_p]
        self.off_R = [mx.array(R) for R in off_R]
        self.axes_mx = [mx.array(np.asarray(a, np.float32)) for a in self.axes]
        self.off_identity = [bool(np.array_equal(R, np.eye(3, dtype=np.float32))) for R in off_R]
        self.tracked_idx = list(meta["kin_tracked_idx"])
        self.num_bodies = len(self.parent)
        self.num_joints = len(meta["joint_names"])
        self.q_lo_np = np.asarray(meta["q_lo"], np.float32)
        self.q_hi_np = np.asarray(meta["q_hi"], np.float32)
        self.q_lo, self.q_hi = mx.array(self.q_lo_np), mx.array(self.q_hi_np)

    def forward(self, p, R, v, w, q, qd, all_bodies: bool = False):
        """Tracked-body world positions (..., n, 3), rotations (..., n, 3, 3), link-origin linear and angular velocities
        (..., n, 3), from the root state (p, R, v, w) and joint q, qd (..., J). all_bodies returns every body."""
        n = self.num_bodies
        J = self.num_joints
        qs, qds = [q[..., j : j + 1] for j in range(J)], [qd[..., j : j + 1] for j in range(J)]  # (..., 1) slices
        P, Rs, V, W = [None] * n, [None] * n, [None] * n, [None] * n
        for i in range(n):
            par = self.parent[i]
            if par < 0:
                P[i], Rs[i], V[i], W[i] = p, R, v, w
                continue
            j = self.joint_idx[i]
            Rp = Rs[par]
            pi = P[par] + mv3(Rp, self.off_p[i])
            R_off = Rp if self.off_identity[i] else mm3(Rp, self.off_R[i])  # skip the product for identity offsets
            Ri = mm3(R_off, axis_angle_to_mat(self.axes[i], qs[j]))
            wi = W[par] + mv3(R_off, self.axes_mx[i]) * qds[j]
            vi = V[par] + cross(W[par], pi - P[par])
            P[i], Rs[i], V[i], W[i] = pi, Ri, vi, wi
        idx = range(n) if all_bodies else self.tracked_idx
        return (mx.stack([P[i] for i in idx], axis=-2), mx.stack([Rs[i] for i in idx], axis=-3),
                mx.stack([V[i] for i in idx], axis=-2), mx.stack([W[i] for i in idx], axis=-2))  # fmt: skip
