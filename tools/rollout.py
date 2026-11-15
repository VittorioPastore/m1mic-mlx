"""Play a checkpoint on one clip, without randomization or noise, and save the trajectory.

    python tools/rollout.py <ckpt_*.npz> --clip dance1_subject2 --start 60 --seconds 12 --out rollout.npz
    python tools/rollout.py <ckpt_*.npz> --task velocity --cmd 1.0,0,0.5
    python tools/rollout.py <ckpt_*.npz> --task velocity --cmd_schedule "0:0,0,0;2:1,0,0;6:1,0,0.8"

The npz holds the robot's qpos, the reference qpos (``ref``) and the reference frame at each 50 Hz control step.
tools/render_rollout.py turns it into a GIF.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from m1mic.config import floats  # noqa: E402
from m1mic.evaluate import rollout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--clip", default="dance1_subject2")
    ap.add_argument("--start", type=float, default=60.0, help="start time in the clip (s)")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--motion_dir", default=None, help="default: the run's own")
    ap.add_argument("--out", default="rollout.npz")
    ap.add_argument("--task", default="motion", choices=("motion", "velocity"))
    ap.add_argument("--cmd", default=None, help="--task velocity: a fixed command vx,vy,yaw_rate (default: the clip's)")
    ap.add_argument("--cmd_schedule", default=None, help='--task velocity: "t:vx,vy,yaw_rate;..." (seconds), each held from t on')
    args = ap.parse_args()
    cmd = None if args.cmd is None else floats(args.cmd)
    if args.cmd_schedule:
        cmd = [(float(t), floats(c)) for t, c in (part.split(":") for part in args.cmd_schedule.split(";"))]
    r = rollout(args.ckpt, args.clip, args.start, args.seconds, args.motion_dir, args.task, cmd)
    np.savez(args.out, **r, clip=args.clip)
    print(f"{args.clip} from {args.start:.1f} s: {len(r['f']) - 1} steps, alive to the end: {bool(r['alive'][-1])} -> {args.out}")


if __name__ == "__main__":
    main()
