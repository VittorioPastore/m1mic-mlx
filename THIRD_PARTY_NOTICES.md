# Third-party notices

## XMimic

`m1mic/` is a port of XMimic (https://github.com/nico-bohlinger/xmimic) to MLX, used under XMimic's MIT License
(Copyright (c) 2026 Nico Bohlinger), whose text is the one in [LICENSE](LICENSE). XMimic in turn derives from the
projects below, and their notices apply to this repository as they do to XMimic.

## BlinkMimic

XMimic reimplements BlinkMimic (https://github.com/fuyichen0725/blinkmimic, commit `d264658`) in JAX. Much of
XMimic (simulator, task, motion library, kinematics, replay buffer, networks, optimizers, evaluation protocol) is a
port of BlinkMimic's code, used under its MIT License:

```
MIT License

Copyright (c) 2026 Fuyi Chen
Copyright (c) 2026 Huy-Quang-Dao (InstantTrack, from which this software is derived)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## InstantTrack and supermimic_vibed

BlinkMimic is derived from InstantTrack (https://github.com/Huy-Quang-Dao/instantTrack), which started from
supermimic_vibed (https://github.com/zita-ch/supermimic_vibed). The parts of XMimic ported from BlinkMimic include
portions of their code, used under their MIT Licenses:

```
MIT License

Copyright (c) 2026 Huy-Quang-Dao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## SuperMimic

The velocity-command task (`--vel_share`: the velocity encoder in `nets.py`, `Motion.vel_command` in `motion.py`, the
slots and latent loss in `algorithm.py`) is a port of SuperMimic's multi-encoder policy and velocity task
(https://github.com/HybridRobotics/supermimic, commit `5efbf11`: `src/supermimic/models.py` `MultiEncoderAE`,
`src/supermimic/tasks/task_spec.py` `make_velocity_task` and `VelocityTaskCfg`), used under the Apache License 2.0, a
copy of which is in [`licenses/SuperMimic-Apache-2.0.txt`](licenses/SuperMimic-Apache-2.0.txt). The RL training of that
encoder (the velocity reward and fall termination in `mdp.py`, the command resampler, the velocity critic and the
decoder's stop-gradient in `algorithm.py`, the step test in `evaluate.py`) is not in SuperMimic's released code. It is
implemented from the paper (arXiv:2610.28480, section 4.5 and Appendices B and G).

```
Copyright 2026 The SuperMimic Authors.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with
the License. Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
```

Changes from SuperMimic's code: rewritten in MLX inside XMimic's networks and BPTT learner; one velocity encoder beside the
motion encoder instead of a dictionary of task encoders; the command's mean velocity is the displacement over the window
and its window is clamped to the clip, not to the whole library; the velocity rows see an identity reference orientation
in XMimic's observation, which has one; the reference skip of the decoder is switched off when the task is on.

## Unitree G1 description

`assets/g1/scene.xml` and `assets/g1/meta.json` were exported from BlinkMimic's G1 scene (compiled by mjlab,
Apache-2.0), which is built from Unitree's G1 description. `tools/download_data.sh` downloads the G1 meshes; they are not
included here. Unitree's description is distributed under the BSD 3-Clause License:

```
BSD 3-Clause License

Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Figure

The laptop in `figures/m1mic-mlx.png` is the MacBook Pro model of the "floating laptop" example of pmndrs/examples
(https://github.com/pmndrs/examples, `examples/floating-laptop/src/mac-draco.glb`), a repository under the MIT License
(Copyright (c) 2024 Poimandres). That repository does not name the model's own author.

## Data

The motion clips are LAFAN1 (Ubisoft La Forge; F. G. Harvey, M. Yurick, D. Nowrouzezahrai, C. Pal, "Robust Motion
In-betweening", ACM TOG / SIGGRAPH 2020), licensed CC BY-NC-ND 4.0, retargeted to the G1 in the Hugging Face dataset
`lvhaidong/LAFAN1_Retargeting_Dataset`. They are not redistributed here. `figures/m1mic-mlx.png`
(its robots are posed from rollouts on LAFAN1 clips) and any trained checkpoint derive from LAFAN1 and are for
non-commercial use only.

## Dependencies

MLX (MIT), MuJoCo (Apache-2.0), NumPy and Pillow are installed from their
own distributions under their own licenses and are not redistributed here.
