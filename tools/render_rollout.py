"""Render a rollout from tools/rollout.py as a GIF: the policy on the left, the reference motion on the right.

    python tools/render_rollout.py rollout.npz --out rollout.gif

Needs pillow. Velocity rollouts have no reference, so only the policy is drawn.
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from m1mic.model import load_full_model  # noqa: E402


def frames(model, qpos, width, height, cam):
    data = mujoco.MjData(model)
    with mujoco.Renderer(model, height, width) as r:
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            cam.lookat[:] = data.qpos[:3]
            r.update_scene(data, cam)
            yield r.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollout")
    ap.add_argument("--out", default="rollout.gif")
    ap.add_argument("--fps", type=int, default=25, help="the rollout is 50 Hz; frames are dropped to match")
    ap.add_argument("--width", type=int, default=480, help="per panel")
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    z = np.load(args.rollout)
    model, _ = load_full_model()
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 3.0, 135, -15
    skip = max(round(50 / args.fps), 1)
    panels = [frames(model, z[k][::skip], args.width, args.height, cam) for k in ("qpos", "ref") if k in z.files]
    images = [Image.fromarray(np.concatenate(row, axis=1)) for row in zip(*panels)]
    images[0].save(args.out, save_all=True, append_images=images[1:], duration=1000 // args.fps, loop=0)
    print(f"{len(images)} frames -> {args.out}")


if __name__ == "__main__":
    main()
