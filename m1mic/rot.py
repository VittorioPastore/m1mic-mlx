"""Batched rotations in MLX. Port of XMimic's rot.py. Quaternions are wxyz, matrices act on column vectors.

Same formulas as the JAX version, written to avoid reductions over inner axes, which are slow on Metal.
"""

import mlx.core as mx


def tr3(A):
    return mx.swapaxes(A, -1, -2)


def _cotangent(x, shape):
    """Reduce a custom gradient x to the operand's ``shape`` by summing over broadcast axes."""
    if x.shape != tuple(shape):
        nd = x.ndim - len(shape)
        if nd:
            x = x.sum(axis=tuple(range(nd)))
        axes = tuple(i for i, (a, b) in enumerate(zip(x.shape, shape)) if b == 1 and a != 1)
        if axes:
            x = x.sum(axis=axes, keepdims=True)
    # fusion barrier: without it mx.compile (MLX 0.32) merges many gradient kernels past Metal's argument limit
    # ("Too many inputs/outputs fused in the Metal Compiled primitive")
    return mx.contiguous(x)


# 3x3 products as sums of three column-row products: mx.compile fuses them into one kernel, 10-20x faster on Metal
# than a batched matmul or a broadcast product summed over an inner axis
def _mm(A, B):
    return A[..., :, 0:1] * B[..., 0:1, :] + A[..., :, 1:2] * B[..., 1:2, :] + A[..., :, 2:3] * B[..., 2:3, :]


def _mv(A, v):
    return A[..., :, 0] * v[..., 0:1] + A[..., :, 1] * v[..., 1:2] + A[..., :, 2] * v[..., 2:3]


def _cross(a, b):
    ax, ay, az = a[..., 0:1], a[..., 1:2], a[..., 2:3]
    bx, by, bz = b[..., 0:1], b[..., 1:2], b[..., 2:3]
    return mx.concatenate([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


# mm3, mv3 and cross carry their own vjps, written as the same products, so the backward pass has no reductions
# (about 5x faster at 16k rows)
@mx.custom_function
def mm3(A, B):
    """Matrix product of (..., 3, 3) arrays (batch dimensions broadcast)."""
    return _mm(A, B)


@mm3.vjp
def _mm3_vjp(primals, cotangent, output):
    A, B = primals
    return _cotangent(_mm(cotangent, tr3(B)), A.shape), _cotangent(_mm(tr3(A), cotangent), B.shape)


@mx.custom_function
def mv3(A, v):
    """Matrix-vector product of (..., 3, 3) and (..., 3) arrays (batch dimensions broadcast)."""
    return _mv(A, v)


@mv3.vjp
def _mv3_vjp(primals, cotangent, output):
    A, v = primals
    return _cotangent(cotangent[..., :, None] * v[..., None, :], A.shape), _cotangent(_mv(tr3(A), cotangent), v.shape)


@mx.custom_function
def cross(a, b):
    """Cross product of (..., 3) arrays (batch dimensions broadcast)."""
    return _cross(a, b)


@cross.vjp
def _cross_vjp(primals, cotangent, output):
    a, b = primals
    return _cotangent(_cross(b, cotangent), a.shape), _cotangent(_cross(cotangent, a), b.shape)


def _elements(m):
    """The nine elements of (..., 3, 3) matrices, row-major, each of shape (..., 1)."""
    # slices, not mx.split: mx.compile (MLX 0.32) fails with "unordered_map::at: key not found" when the same
    # array is split twice
    m = m.reshape(m.shape[:-2] + (9,))
    return [m[..., k : k + 1] for k in range(9)]


def _matrix(elements):
    """(..., 3, 3) matrices from nine row-major elements of shape (..., 1)."""
    m = mx.concatenate(elements, axis=-1)
    return m.reshape(m.shape[:-1] + (3, 3))


def _norm(v):
    """Euclidean norm over the last axis (kept). The gradient at zero is zero, not NaN."""
    return mx.sqrt(mx.maximum((v * v).sum(-1, keepdims=True), 1e-30))


def normalize(v, eps=1e-12):
    return v / mx.maximum(_norm(v), eps)


def quat_to_mat(q):
    q = normalize(q)
    w, x, y, z = q[..., 0:1], q[..., 1:2], q[..., 2:3], q[..., 3:4]
    return _matrix([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ])  # fmt: skip


def mat_to_quat(m):
    """Matrix -> quaternion (wxyz, w >= 0) using the largest-diagonal branch, as in BlinkMimic."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = _elements(m)
    qw = mx.sqrt(mx.maximum(1 + m00 + m11 + m22, 1e-10)) / 2
    qx = mx.sqrt(mx.maximum(1 + m00 - m11 - m22, 1e-10)) / 2
    qy = mx.sqrt(mx.maximum(1 - m00 + m11 - m22, 1e-10)) / 2
    qz = mx.sqrt(mx.maximum(1 - m00 - m11 + m22, 1e-10)) / 2
    cands = [
        mx.concatenate([qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)], axis=-1),
        mx.concatenate([(m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)], axis=-1),
        mx.concatenate([(m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)], axis=-1),
        mx.concatenate([(m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz], axis=-1),
    ]
    idx = mx.stop_gradient(mx.argmax(mx.concatenate([qw, qx, qy, qz], axis=-1), axis=-1, keepdims=True))
    q = mx.where(idx == 0, cands[0], mx.where(idx == 1, cands[1], mx.where(idx == 2, cands[2], cands[3])))
    q = normalize(q)
    return q * mx.where(q[..., :1] < 0, -1.0, 1.0)


def axis_angle_to_mat(axis, angle):
    """Rodrigues formula for a fixed unit axis (three Python floats) and angles (..., 1) -> (..., 3, 3)."""
    x, y, z = axis
    c, s = mx.cos(angle), mx.sin(angle)
    C = 1 - c
    return _matrix([
        c + x * x * C, x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, c + y * y * C, y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, c + z * z * C,
    ])  # fmt: skip


def skew(v):
    x, y, z = v[..., 0:1], v[..., 1:2], v[..., 2:3]
    o = mx.zeros_like(x)
    return _matrix([o, -z, y, z, o, -x, -y, x, o])


def so3_exp(v):
    """Rotation vector (..., 3) -> matrix, smooth at zero. I + a K + b K^2 with K = skew(v), written by elements."""
    x, y, z = v[..., 0:1], v[..., 1:2], v[..., 2:3]
    theta2 = x * x + y * y + z * z
    theta = mx.sqrt(theta2 + 1e-12)
    small = theta2 < 1e-8
    a = mx.where(small, 1 - theta2 / 6, mx.sin(theta) / theta)
    b = mx.where(small, 0.5 - theta2 / 24, (1 - mx.cos(theta)) / (theta2 + 1e-12))
    return _matrix([
        1 - b * (y * y + z * z), b * x * y - a * z, b * x * z + a * y,
        b * x * y + a * z, 1 - b * (x * x + z * z), b * y * z - a * x,
        b * x * z - a * y, b * y * z + a * x, 1 - b * (x * x + y * y),
    ])  # fmt: skip


def quat_to_rotvec(q):
    q = q * mx.where(q[..., :1] < 0, -1.0, 1.0)
    w, v = q[..., :1], q[..., 1:]
    sin_half = _norm(v)
    half = mx.arctan2(sin_half, w)
    scale = mx.where(sin_half < 1e-6, 2.0 / mx.maximum(w, 1e-6), 2 * half / (sin_half + 1e-12))
    return v * scale


def so3_log(R):
    return quat_to_rotvec(mat_to_quat(R))


def rot_angle_sq(Ra, Rb):
    """Squared geodesic angle between rotations (acos of the trace, clamped)."""
    p = Ra * Rb
    tr = p.reshape(p.shape[:-2] + (9,)).sum(-1)
    cos = mx.clip((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return mx.arccos(cos) ** 2


def yaw_mat(R):
    """Heading (yaw-only) rotation of R, from the projected x axis."""
    e = _elements(R)
    fx, fy = e[0], e[3]
    n = mx.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / n, fy / n
    o, one = mx.zeros_like(c), mx.ones_like(c)
    return _matrix([c, -s, o, s, c, o, o, o, one])


def mat_to_6d(R):
    """6D rotation: the first two columns flattened row-major, as in BlinkMimic."""
    return R[..., :, :2].reshape(R.shape[:-2] + (6,))
