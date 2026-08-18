# RoboCasa online evaluation

The RoboCasa v0.2 simulator is installed locally without modifying the
training environment:

- training environment: `bsp-simple` (unchanged)
- online evaluation environment: `bsp-robocasa`
- simulator source and assets: `third_party/robocasa-cosmos-policy`
- RoboCasa / robosuite / MuJoCo: `0.2.0 / 1.5.1 / 3.2.6`

The evaluator uses the Human-50 observation and action contract:

- three vertically corrected 224 x 224 RGB views
- two observation steps, ordered oldest to newest
- `ee_pos`, axis-angle `ee_ori`, and two gripper joint positions
- 7D delta EEF action expanded to PandaMobile's 12D action by appending
  `[0, 0, 0, 0, -1]`
- the fixed delta `OSC_POSE` controller used by the public NVIDIA RoboCasa
  evaluation
- five test layout/style pairs and object split B

## One rollout

Run from the repository root:

```bash
conda run -n bsp-robocasa python bspline_policy/eval_robocasa.py \
  --checkpoint bspline_policy/data/outputs/drifting_bspline_robocasa/20260814_1514_human50_seed42/checkpoints/latest.ckpt \
  --output-dir bspline_policy/data/outputs/robocasa_eval/latest_seed195 \
  --device cuda:0 \
  --episodes 1 \
  --video-episodes 1 \
  --max-steps 500 \
  --seed 195 \
  --policy-seed 195
```

The command writes `eval_log.json`, per-episode details in
`robocasa_episodes.json`, and rollout MP4 files under `media/`. Output
directories must be new, preventing an evaluation from overwriting an older
result.

## NVIDIA Human-50 reference evaluation

Use 50 episodes for each of seeds 195, 196, and 197, with a different output
directory for every seed. For example, change the one-rollout command to
`--episodes 50 --video-episodes 3`. Aggregate the three
`test/success_rate` values only after all three runs finish.

These seeds and scene scheduling follow the public NVIDIA evaluation for the
Human-50 data mirror. The B-spline paper does not publish its exact RoboCasa
evaluation seeds, so results should be labelled with this protocol rather than
claimed as an exact reproduction of the paper's hidden setup.

The CLI sets `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, and
`NUMBA_DISABLE_JIT=1` inside its own process. The last variable avoids a numba
cache problem in cloned Conda environments; it does not change `bsp-simple`.

RoboCasa v0.2 declares old exact NumPy and numba pins that do not support the
Python version used by the trained checkpoint. The isolated evaluator retains
the checkpoint-compatible NumPy/numba versions and disables numba JIT. This is
also compatible with the environment behavior exercised by the smoke tests.
