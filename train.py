"""Train a G1 motion tracker on a Mac.

    python train.py --log_dir runs/lafan1
    python train.py --clips walk,run --log_dir runs/loco    # only clips whose names start with walk or run

Every field of m1mic/config.py is a flag. Checkpoints, train_log.csv and args.json go to --log_dir.
"""

from m1mic.config import parse_args
from m1mic.runtime import run

if __name__ == "__main__":
    run(parse_args())
