"""Run options. Every field is also a flag of train.py (--name value).

The algorithm and its hyperparameters are BlinkMimic's PRESET[main]. Only the sizes differ: the published run uses
4096 worlds and batches of 2048 on an H100, the defaults here are a quarter of that, about one iteration per second
on an M1 Pro. Published size: --num_envs 4096 --wm_batch 2048 --pi_batch 2048 --eprobe_max 1536.
"""

from __future__ import annotations

import argparse
import dataclasses


@dataclasses.dataclass
class Config:
    # ---- algorithm
    num_envs: int = 1024  # simulated worlds (published: 4096)
    iterations: int = 3500
    buffer_steps: int = 1000  # replay buffer length in control steps
    utd: float = 0.4  # update-to-data ratio: 1 / utd collection steps per iteration
    horizon: int = 32  # imagined steps per policy update
    goal_frames: int = 10  # future reference frames in the policy's goal
    hist: int = 10  # observation history of the policy
    wm_window: int = 8  # open-loop steps in the world-model loss
    wm_batches: int = 5  # world-model updates per iteration
    wm_batch: int = 512  # (published: 2048)
    pi_batch: int = 512  # imagined rollouts per policy update (published: 2048)
    lr: float = 4e-4
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    wm_hidden: str = "1024,1024,1024"
    enc_hidden: str = "2048,1024,512,256"
    dec_hidden: str = "2048,1024,512,256"
    fail_reset: float = 0.3  # share of resets drawn near recent failures
    pi_beta1: float = 0.9
    pi_clip_value: float = 0.1  # elementwise policy-gradient clip
    pi_ema: float = 0.995  # policy weight average (the average is what gets deployed)
    warmup_steps: int = 64  # collection steps before the first update
    contact_reward: float = 0.25
    wm_mirror: float = 0.5  # share of each world-model batch mirrored left-right
    pi_lr_mult: float = 4.0  # policy LR multiplier before the decay trigger
    lr_decay_ret: float = 0.75  # decay starts when the mean episode return reaches this
    lr_decay_len: int = 300
    lr_decay_by: int = 1400  # or at this iteration
    lr_decay_iters: int = 250_000  # cosine LR schedule length
    # ---- evaluation-guided resets (BlinkMimic --eprobe_*): the averaged policy is swept over the clips
    #      and the failures it finds get more training resets
    eprobe_share: float = 0.15  # share of resets drawn from the focus distribution (0 disables the probes)
    eprobe_tile: int = 300  # one probe world per tile of this many frames
    eprobe_max: int = 384  # max probe worlds per sweep (published: 1536)
    eprobe_lead_min: int = 200  # probes start a random lead before their tile
    eprobe_lead_max: int = 450
    eprobe_every: int = 125  # iterations between sweeps
    eprobe_start: int = 1000  # first sweep
    eprobe_win: int = 10  # failures spread over this many bins before them
    eprobe_decay: float = 0.7  # decay of the failure counts per sweep
    eprobe_async: int = 20  # sweep steps per training iteration while a sweep runs
    # ---- velocity-command task (SuperMimic's velocity encoder)
    vel_share: float = 0.0  # share of worlds and imagined rollouts on the velocity task (0: off)
    vel_window: int = 25  # frames averaged for a clip's reference command
    vel_lo: str = "-1.5,-1.32,-3.35"  # lower limits of vx (m/s), vy (m/s), yaw rate (rad/s)
    vel_hi: str = "2.0,1.32,3.35"  # upper limits
    vel_ref_prob: float = 0.5  # chance a resampled command follows the clip, else a held random one
    vel_zero_prob: float = 0.1  # chance a held command is zero (standing)
    vel_resample: int = 250  # control steps between command resamples (5 s)
    vel_latent_bc: float = 0.1  # weight of the latent matching loss
    vel_enc_hidden: str = "256,256"  # hidden widths of the velocity encoder
    pi_ref_skip: int = 1  # add the next reference pose to the decoder output; must be 0 with vel_share
    # ---- run
    log_every: int = 50  # iterations per log line
    ckpt_every: int = 250  # a checkpoint is 36 MB
    seed: int = 0
    motion_dir: str = "data/lafan1_g1"
    clips: str = ""  # clip-name prefixes to train on, e.g. "walk,run" (default: all)
    log_dir: str = ""  # default: runs/<YYYY-mm-dd_HH-MM-SS>
    max_minutes: float = 0.0  # stop after this much training time (0: run all iterations)
    # ---- machine
    threads: int = 0  # physics threads (0: one per CPU core)
    compile: int = 1  # mx.compile the update functions
    cache_gb: float = 1.0  # cap on MLX's buffer cache; its default is nearly all memory (0: keep it)


def floats(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(","))


def widths(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(","))


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description="Train a G1 motion tracker with MLX (options: m1mic/config.py).")
    for f in dataclasses.fields(Config):
        p.add_argument("--" + f.name, type=type(f.default), default=f.default)
    cfg = Config(**vars(p.parse_args(argv)))
    assert 0.0 <= cfg.vel_share < 1.0, "--vel_share must be in [0, 1): the motion encoder is the latent loss's anchor"
    assert cfg.vel_share == 0 or not cfg.pi_ref_skip, (
        "--vel_share needs --pi_ref_skip 0: the velocity command has no reference pose to add to the shared decoder's output")
    lo, hi = floats(cfg.vel_lo), floats(cfg.vel_hi)
    assert len(lo) == 3 and len(hi) == 3 and all(a <= 0 <= b for a, b in zip(lo, hi)), (
        "--vel_lo and --vel_hi take 3 values each, with lo <= 0 <= hi (the zero command is in the range)")
    assert 0 <= cfg.vel_ref_prob <= 1 and 0 <= cfg.vel_zero_prob <= 1 and cfg.vel_resample > 0
    if cfg.vel_share > 0:
        for n, what in ((cfg.num_envs, "--num_envs"), (cfg.pi_batch, "--pi_batch")):
            assert 0 < round(n * cfg.vel_share) < n, f"--vel_share {cfg.vel_share} leaves a slot of {what} empty"
    if not cfg.log_dir:
        import time

        cfg.log_dir = "runs/" + time.strftime("%Y-%m-%d_%H-%M-%S")
    return cfg
