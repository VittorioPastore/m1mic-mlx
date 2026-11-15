"""The Unitree G1 tracking scene as a MuJoCo model.

assets/g1/scene.xml and meta.json come from BlinkMimic's mjlab scene. The XML rounds floats to about 7 digits, so the
rounded fields are restored from the exact values in meta.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "g1"
SIM_OPTIONS = ("timestep", "integrator", "cone", "solver", "iterations", "tolerance", "ls_iterations", "ls_tolerance",
               "ccd_iterations", "impratio")  # fmt: skip


def load_meta(path: Path = ASSETS) -> dict:
    """Joint and body order, actuators, default pose, action scale, domain randomization, solver options."""
    return json.loads((Path(path) / "meta.json").read_text())


def load_full_model(path: Path = ASSETS) -> tuple[mujoco.MjModel, dict]:
    """(MjModel with BlinkMimic's solver options and exact parameters, metadata), with the full meshes."""
    path = Path(path)
    if not (path / "assets" / "meshes").is_dir():
        raise FileNotFoundError(f"G1 meshes not found in {path / 'assets' / 'meshes'}: run tools/download_data.sh first")
    meta = load_meta(path)
    m = mujoco.MjModel.from_xml_path(str(path / "scene.xml"))
    for name in SIM_OPTIONS:
        setattr(m.opt, name, meta["sim"][name])
    for field, value in meta["exact"].items():
        getattr(m, field)[:] = np.asarray(value, dtype=np.float64).reshape(getattr(m, field).shape)
    return m, meta


def load_model(path: Path = ASSETS, light: bool = True) -> tuple[mujoco.MjModel, dict]:
    """load_full_model, with every mesh replaced by its convex hull when ``light``.

    MuJoCo collides meshes through their hulls anyway, so the dynamics match (about 1e-6 per control step) and the
    model is 15 MB instead of 75 MB. Each physics thread keeps its own copy, so the size matters.
    """
    path = Path(path)
    full, meta = load_full_model(path)
    if not light:
        return full, meta
    spec = mujoco.MjSpec.from_file(str(path / "scene.xml"))
    R = np.zeros(9)
    for mesh in spec.meshes:
        mid = mujoco.mj_name2id(full, mujoco.mjtObj.mjOBJ_MESH, mesh.name)
        va, vn = full.mesh_vertadr[mid], full.mesh_vertnum[mid]
        verts = full.mesh_vert[va : va + vn]
        ga = full.mesh_graphadr[mid]
        if ga >= 0:  # colliding mesh: the hull's graph lists its vertices
            nh = full.mesh_graph[ga]
            hull = verts[full.mesh_graph[ga + 2 + nh : ga + 2 + 2 * nh]]
        else:  # visual-only mesh has no hull: use its bounding box
            lo, hi = verts.min(0), verts.max(0)
            hull = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        # compiled meshes are centred and aligned to their inertia frame: undo that
        mujoco.mju_quat2Mat(R, full.mesh_quat[mid])
        mesh.file = ""
        mesh.uservert = ((hull @ R.reshape(3, 3).T + full.mesh_pos[mid]) / full.mesh_scale[mid]).astype(np.float64).ravel()
    m = spec.compile()
    assert (m.nq, m.nv, m.nu, m.ngeom, m.nbody) == (full.nq, full.nv, full.nu, full.ngeom, full.nbody)
    m.opt = full.opt
    for field in meta["exact"]:
        getattr(m, field)[:] = getattr(full, field)
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "geom_friction"):
        assert np.array_equal(getattr(m, field), getattr(full, field)), field
    return m, meta
