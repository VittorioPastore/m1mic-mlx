"""Reference motion library in MLX.

G1-retargeted LAFAN1 clips (30 fps CSV) are resampled to the 50 Hz control rate. Velocities and tracked-body poses
are derived from the stored root and joint trajectories. ``*_np`` arrays mirror the frame bookkeeping for host code.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os

import mlx.core as mx
import numpy as np

from .rot import mat_to_6d, mm3, mv3, quat_to_mat, so3_log, tr3, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50
GOAL_Q_OFFSET = 15  # index of q(f+1) in the goal vector, after dp 3, rot 6, v 3, w 3
Q_MAX = 3.2  # rad, joint angles are int16 fixed point over [-Q_MAX, Q_MAX] (step 9.8e-5 rad)
CACHE_VERSION = 2


def load_lafan1_csv(path: str):
    """30 fps rows of root position, root quaternion (xyzw) and 29 joint angles -> (p, quat wxyz, q, fps)."""
    raw = np.loadtxt(path, delimiter=",")
    p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
    return p, np.concatenate((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1), j, 30.0


def _slerp_np(q0, q1, t):
    """BlinkMimic's quat_slerp in float32 numpy."""
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.minimum(np.abs(dot), np.float32(1.0))
    theta = np.arccos(dot)
    s = np.sin(theta)
    t = t[..., None]
    w0 = np.where(s < 1e-6, 1 - t, np.sin((1 - t) * theta) / (s + np.float32(1e-12)))
    w1 = np.where(s < 1e-6, t, np.sin(t * theta) / (s + np.float32(1e-12)))
    q = w0 * q0 + w1 * q1
    return (q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), np.float32(1e-12))).astype(np.float32)


def _read_clip(f):
    p, quat, j, fps = load_lafan1_csv(f)
    p, quat, j = (x.astype(np.float32) for x in (p, quat, j))
    T_src = p.shape[0]
    dur = (T_src - 1) / fps
    n = int(np.floor(dur / CONTROL_DT)) + 1
    t = np.arange(n, dtype=np.float32) * np.float32(CONTROL_DT) * np.float32(fps)
    i0 = np.minimum(np.floor(t).astype(np.int64), T_src - 2)
    a = np.clip(t - i0.astype(np.float32), 0, 1)
    p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
    j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
    quat = _slerp_np(quat[i0], quat[i0 + 1], a)
    return p.astype(np.float32), quat.astype(np.float32), j.astype(np.float32)


def _read_clips(files, workers: int):
    """Resample each clip to the control rate: (root_p, root_quat wxyz, q) float32 per clip.

    workers > 1 uses processes, so the calling script needs an ``if __name__ == "__main__"`` guard.
    """
    if workers <= 1 or len(files) < 64:
        return [_read_clip(f) for f in files]
    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_read_clip, files, chunksize=64))


class _Lazy:
    """Array-like whose ``x[f]`` is computed on demand."""

    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, f):
        return self.fn(f if isinstance(f, mx.array) else mx.array(np.asarray(f, np.int32)))


TABLES = ("root_R", "q", "root_v", "root_w", "qd", "goal_static", "body_p", "body_R", "body_v", "body_w")
TABLE_BYTES = lambda J, nb: 4 * (9 + J + 6 + J + (6 + 2 * J) + nb * 18)  # noqa: E731  per frame


class Motion:
    """All clips concatenated along one frame axis.

    Stored per frame: root position, root quaternion (wxyz), int16 joint angles (see Q_MAX), reset bin and reference
    foot contacts. Velocities are central differences inside a clip, one-sided at its ends. Tracked bodies come from
    forward kinematics. Derived fields such as ``motion.body_p[f]`` index like arrays, with f an integer MLX array of
    any shape. ``build_tables`` precomputes them for every frame (about 1.6 kB per frame), which keeps the reference
    kinematics out of the training graph. Without tables they are computed when indexed.
    """

    def __init__(self, root_p, root_quat, q_i16, clip_start, clip_end, bin_clip, bin_start, frame_bin, contact_u8, names, kin):
        self.root_p, self.root_quat, self.q_i16 = mx.array(root_p), mx.array(root_quat), mx.array(q_i16)
        self.clip_start_np, self.clip_end_np = np.asarray(clip_start, np.int32), np.asarray(clip_end, np.int32)
        self.bin_clip_np, self.bin_start_np = np.asarray(bin_clip, np.int32), np.asarray(bin_start, np.int32)
        self.frame_bin_np = np.asarray(frame_bin, np.int32)
        self.clip_id_np = self.bin_clip_np[self.frame_bin_np]  # clip of every frame
        self.clip_start, self.clip_end = mx.array(self.clip_start_np), mx.array(self.clip_end_np)
        self.bin_clip, self.bin_start = mx.array(self.bin_clip_np), mx.array(self.bin_start_np)
        self.frame_bin = mx.array(self.frame_bin_np)
        self.frame_clip = mx.array(self.clip_id_np)
        self.contact_u8 = mx.array(np.asarray(contact_u8, np.uint8))
        self.names = tuple(names)
        self.num_frames, self.num_bins = int(self.frame_bin_np.shape[0]), int(self.bin_start_np.shape[0])
        self.num_joints = int(q_i16.shape[-1])
        self.kin = kin
        self.tab = {}  # filled by build_tables

    # derived quantities
    def _root_R(self, f):
        return quat_to_mat(self.root_quat[f])

    def _q(self, f):
        return self.q_i16[f].astype(mx.float32) * (Q_MAX / 32767)

    def _vel(self, f):
        c = self.frame_clip[f]
        fp = mx.minimum(f + 1, self.clip_end[c])
        fm = mx.maximum(f - 1, self.clip_start[c])
        dt = (mx.maximum(fp - fm, 1).astype(mx.float32) * CONTROL_DT)[..., None]
        v = (self.root_p[fp] - self.root_p[fm]) / dt
        w = so3_log(mm3(self._root_R(fp), tr3(self._root_R(fm)))) / dt
        qd = (self._q(fp) - self._q(fm)) / dt
        return v, w, qd

    def _bodies(self, f):
        v, w, qd = self._vel(f)
        return self.kin.forward(self.root_p[f], self._root_R(f), v, w, self._q(f), qd)

    def _goal_static(self, f):
        """Part of a goal frame independent of the current frame: root-frame velocities, q and 0.1 qd."""
        v, w, qd = self._vel(f)
        Rft = tr3(self._root_R(f))
        return mx.concatenate([mv3(Rft, v), mv3(Rft, w), self._q(f), qd * 0.1], axis=-1)

    def build_tables(self, chunk: int = 32768):
        """Precompute the derived quantities for every frame, ``chunk`` frames at a time."""
        parts = {k: [] for k in TABLES}
        for s in range(0, self.num_frames, chunk):
            f = mx.arange(s, min(s + chunk, self.num_frames), dtype=mx.int32)
            v, w, qd = self._vel(f)
            R, q = self._root_R(f), self._q(f)
            bp, bR, bv, bw = self.kin.forward(self.root_p[f], R, v, w, q, qd)
            Rft = tr3(R)
            row = {"root_R": R, "q": q, "root_v": v, "root_w": w, "qd": qd, "body_p": bp, "body_R": bR, "body_v": bv, "body_w": bw,
                   "goal_static": mx.concatenate([mv3(Rft, v), mv3(Rft, w), q, qd * 0.1], axis=-1)}  # fmt: skip
            mx.eval(row)
            for k in TABLES:
                parts[k].append(np.asarray(row[k]))
        self.tab = {k: mx.array(np.concatenate(parts[k])) for k in TABLES}
        mx.eval(self.tab)

    def _get(self, name, fn):
        return self.tab[name] if name in self.tab else _Lazy(fn)

    root_R = property(lambda self: self._get("root_R", self._root_R))
    q = property(lambda self: self._get("q", self._q))
    clip_id = property(lambda self: self.frame_clip)
    ref_contact = property(lambda self: _Lazy(lambda f: self.contact_u8[f].astype(mx.float32)))
    root_v = property(lambda self: self._get("root_v", lambda f: self._vel(f)[0]))
    root_w = property(lambda self: self._get("root_w", lambda f: self._vel(f)[1]))
    qd = property(lambda self: self._get("qd", lambda f: self._vel(f)[2]))
    body_p = property(lambda self: self._get("body_p", lambda f: self._bodies(f)[0]))
    body_R = property(lambda self: self._get("body_R", lambda f: self._bodies(f)[1]))
    body_v = property(lambda self: self._get("body_v", lambda f: self._bodies(f)[2]))
    body_w = property(lambda self: self._get("body_w", lambda f: self._bodies(f)[3]))

    def vel(self, f):
        """(root_v, root_w, qd) at frames f."""
        if self.tab:
            return self.tab["root_v"][f], self.tab["root_w"][f], self.tab["qd"][f]
        return self._vel(f)

    def bodies(self, f):
        """Tracked-body world positions, rotations, linear and angular velocities at frames f."""
        if self.tab:
            return self.tab["body_p"][f], self.tab["body_R"][f], self.tab["body_v"][f], self.tab["body_w"][f]
        return self._bodies(f)

    def goal_dim(self, K: int) -> int:
        return K * (3 + 6 + 3 + 3 + 2 * self.num_joints)

    def future_frames(self, f, K: int):
        """The K frames after f, held at the clip's last frame."""
        end = self.clip_end[self.frame_clip[f]]
        fut = f[..., None] + mx.arange(1, K + 1, dtype=f.dtype)
        return mx.minimum(fut, end[..., None])

    def goal(self, f, K: int):
        """Policy goal (..., K*(15+2J)) from K future reference frames.

        Per frame: root displacement and 6D rotation in the current heading frame, then root-frame velocities, q
        and 0.1 qd.
        """
        fut = self.future_frames(f, K)
        Ht = tr3(yaw_mat(self.root_R[f]))[..., None, :, :]
        dp = mv3(Ht, self.root_p[fut] - self.root_p[f][..., None, :])
        rot = mat_to_6d(mm3(Ht, self.root_R[fut]))
        static = self.tab["goal_static"][fut] if self.tab else self._goal_static(fut)
        return mx.concatenate([dp, rot, static], axis=-1).reshape(f.shape + (-1,))

    def vel_command(self, f, window: int):
        """Reference velocity command (..., 3) at frames f: heading-frame planar root velocity and world yaw rate,
        averaged over ``window`` frames centred on f and clamped to the clip."""
        if window <= 1:
            v, w, _ = self.vel(f)
        else:
            c, h = self.frame_clip[f], window // 2
            fp = mx.minimum(f + h, self.clip_end[c])
            fm = mx.maximum(f - h, self.clip_start[c])
            dt = (mx.maximum(fp - fm, 1).astype(mx.float32) * CONTROL_DT)[..., None]
            v = (self.root_p[fp] - self.root_p[fm]) / dt
            w = so3_log(mm3(self.root_R[fp], tr3(self.root_R[fm]))) / dt
        v_h = mv3(tr3(yaw_mat(self.root_R[f])), v)
        return mx.concatenate([v_h[..., :2], w[..., 2:3]], axis=-1)

    def ref_state(self, f):
        v, w, qd = self.vel(f)
        return self.root_p[f], self.root_R[f], v, w, self.q[f], qd


def load_motion(motion_dir: str, kin, feet: list[int], margin: float = 0.015, vmax: float = 0.5, quant: float = 0.05,
                cache: bool = True, clips: str = "", tables_gb: float = 2.0) -> Motion:
    """Load the *.csv clips of ``motion_dir`` in sorted order with foot contacts (BlinkMimic --ref_contact_fix).

    clips: comma-separated name prefixes to keep (empty: all).
    tables_gb: build the per-frame tables (Motion.build_tables) if they fit in this much memory.
    Resampled clips and contacts are cached in <motion_dir>/.motion_cache_mlx.npz, keyed on the clip files' sizes
    and mtimes and the contact parameters.
    """
    files = sorted(glob.glob(os.path.join(motion_dir, "*.csv")))
    if clips:
        pre = tuple(p for p in clips.split(",") if p)
        files = [f for f in files if os.path.basename(f).startswith(pre)]
    assert files, f"no clips in {motion_dir}: run tools/download_data.sh first"
    names = tuple(os.path.splitext(os.path.basename(f))[0] for f in files)
    h = hashlib.sha1(json.dumps([CACHE_VERSION, feet, margin, vmax, quant, CONTROL_DT, kin.tracked_idx]).encode())
    for f in files:
        st = os.stat(f)
        h.update(f"{os.path.basename(f)}:{st.st_size}:{st.st_mtime_ns};".encode())
    key = h.hexdigest()
    cache_f = os.path.join(motion_dir, ".motion_cache_mlx.npz")
    z = None
    if cache and os.path.exists(cache_f):
        try:
            z = np.load(cache_f)
            if str(z["key"]) != key:
                z = None
        except (OSError, ValueError, KeyError):
            z = None
    if z is not None:
        root_p, root_quat, q, lens, ref_contact = z["root_p"], z["root_quat"], z["q"], z["lens"], z["ref_contact"]
    else:
        clip_data = _read_clips(files, int(os.environ.get("XMIMIC_LOAD_WORKERS", "1")))
        lens = np.asarray([c[0].shape[0] for c in clip_data])
        root_p, root_quat, q = (np.concatenate([c[i] for c in clip_data]) for i in range(3))
        ref_contact = None
    lens = np.asarray(lens)
    clip_start = np.concatenate(([0], np.cumsum(lens[:-1]))).astype(np.int32)
    clip_end = (clip_start + lens - 1).astype(np.int32)
    clip_id = np.repeat(np.arange(len(lens), dtype=np.int32), lens)
    nbins = np.ceil(lens / BIN_FRAMES).astype(np.int64)
    bin_clip = np.repeat(np.arange(len(lens), dtype=np.int32), nbins)
    bin_start = (clip_start[bin_clip] + (np.arange(int(nbins.sum())) - np.repeat(np.cumsum(nbins) - nbins, nbins)) * BIN_FRAMES).astype(np.int32)
    frame_bin = (np.repeat(np.cumsum(nbins) - nbins, lens) + (np.arange(len(clip_id)) - clip_start[clip_id]) // BIN_FRAMES).astype(np.int32)
    n_clip = int((np.abs(q) > Q_MAX).sum())
    if n_clip:
        print(f"warning: {n_clip} joint angles beyond +-{Q_MAX} rad clipped", flush=True)
    q_i16 = np.round(np.clip(q, -Q_MAX, Q_MAX) * (32767 / Q_MAX)).astype(np.int16)
    zero_contact = np.zeros((len(clip_id), len(feet)), np.uint8)
    mo = Motion(root_p, root_quat, q_i16, clip_start, clip_end, bin_clip, bin_start, frame_bin, zero_contact, names, kin)
    if ref_contact is None:
        # contact: foot below the clip's floor (height quantile) + margin and slower than vmax
        feet_a = mx.array(np.asarray(feet, np.int32))
        CH = 32768
        zs, vs = [], []
        for s in range(0, mo.num_frames, CH):
            f = mx.arange(s, min(s + CH, mo.num_frames), dtype=mx.int32)
            bp, _, bv, _ = mo._bodies(f)
            zz = mx.take(bp, feet_a, axis=-2)[..., 2]
            vxy = mx.take(bv, feet_a, axis=-2)[..., :2]
            vv = mx.sqrt((vxy * vxy).sum(-1))
            mx.eval(zz, vv)
            zs.append(np.asarray(zz)), vs.append(np.asarray(vv))
        zf, vh = np.concatenate(zs), np.concatenate(vs)
        floor = np.zeros_like(zf)
        for c in range(len(lens)):
            a, b = clip_start[c], clip_end[c] + 1
            floor[a:b] = np.quantile(zf[a:b], quant, axis=0)
        ref_contact = ((zf < floor + margin) & (vh < vmax)).astype(np.float32)
        if cache:
            try:
                tmp = cache_f + ".tmp.npz"
                np.savez(tmp, key=key, root_p=root_p, root_quat=root_quat, q=q, lens=lens, ref_contact=ref_contact)
                os.replace(tmp, cache_f)
            except OSError:
                pass
    mo.contact_u8 = mx.array(ref_contact.astype(np.uint8))
    if mo.num_frames * TABLE_BYTES(mo.num_joints, len(kin.tracked_idx)) <= tables_gb * 1e9:
        mo.build_tables()
    return mo
