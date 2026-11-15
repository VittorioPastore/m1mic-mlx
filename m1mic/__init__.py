"""M1mic-MLX: humanoid motion tracking trained on a Mac.

MuJoCo steps the worlds on the CPU, MLX trains the world model, policy and critic on the GPU.

  model, motion, kin, rot    the G1 model, the reference motions and their kinematics
  sim, mdp, buffer           batched CPU simulator, tracking task, replay buffer
  nets, optim                world model, policy and critic; MuonAdam, Adam, policy weight average
  algorithm                  one training iteration
  runtime                    logging, checkpoints, the training loop
  evaluate                   checkpoint evaluation and single rollouts
"""

__version__ = "0.1.0"
