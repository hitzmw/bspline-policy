# B-spline Policy


## Installation

Use the same conda environment as `diffusion_policy` training:

```bash
cd ~/bspline-policy/diffusion_policy
mamba env create -f conda_environment.yaml
conda run -n bsp-simple python -m pip install robomimic==0.2.0 --no-deps
```

And then:

```bash
conda activate bsp-simple
cd ~/bspline-policy/bspline_policy
python -m pip install -e bspline_policy
```

## Data Processing

You can use your own collected YAM demos or download the example dataset used by
the Simple Mobile tutorial:

```bash
cd ~/bspline-policy/real_env
source .venv/bin/activate

cd ~/bspline-policy/real_env/yam_teleop
uv run gdown 1n4iDcV5P52NGHNlB7Ed3EwuDHboADUVf
unzip data.zip
```

Convert the collected episodes into robomimic HDF5:

```bash
cd ~/bspline-policy/real_env
source .venv/bin/activate

cd ~/bspline-policy/real_env/yam_teleop
uv run python convert_to_robomimic_hdf5.py \
  --input-dir data/demos \
  --output-path data/yam-v1.hdf5
```

Move the HDF5 file into the training repo:

```bash
mkdir -p ~/bspline-policy/diffusion_policy/data
cp ~/bspline-policy/real_env/yam_teleop/data/yam-v1.hdf5 \
  ~/bspline-policy/diffusion_policy/data/yam-v1.hdf5
```

For the default YAM config, expected raw action shape is `(T, 7)` and obs keys
are `arm_pos`, `arm_quat`, `gripper_pos`, and `wrist_image`.

## Training

Start a B-spline training run:

```bash
conda activate bsp-simple
cd ~/bspline-policy/bspline_policy

HYDRA_FULL_ERROR=1 WANDB_MODE=offline python train.py \
  --config-name=train_diffusion_unet_real_hybrid_bspline_workspace \
  training.resume=false \
  logging.mode=offline
```

The default config reads:

```text
~/bspline-policy/diffusion_policy/data/yam-v1.hdf5
```

and writes outputs under:

```text
~/bspline-policy/bspline_policy/data/outputs/
```

## Push-T Image Simulation

Point the local Diffusion Policy data directory at an existing dataset:

```bash
ln -s /path/to/diffusion_policy/data \
  ~/bspline-policy/diffusion_policy/data
```

The Push-T B-spline task expects:

```text
diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr
```

The simulation configuration follows the paper setup: cubic B-splines,
fitting tolerance `1.0`, and 16 knot/control-point pairs. On a 16 GB GPU,
use gradient accumulation to preserve an effective batch size of 64:

```bash
conda activate bsp-simple
cd ~/bspline-policy/bspline_policy

HYDRA_FULL_ERROR=1 WANDB_MODE=offline python train.py \
  --config-name=train_diffusion_unet_pusht_image_bspline_workspace \
  training.resume=false \
  dataloader.batch_size=8 \
  val_dataloader.batch_size=8 \
  training.gradient_accumulate_every=8 \
  task.env_runner.n_envs=8 \
  logging.mode=offline
```

### Drifting-BSpline Push-T

The parallel Drifting-BSpline policy replaces DDPM training and iterative
denoising with the faithful Drifting objective and a single UNet evaluation.
It keeps the Push-T B-spline data representation and rollout path unchanged:
cubic splines, fitting tolerance `1.0`, 16 complete
`[knot, control_x, control_y]` rows, monotonic knot projection, and decoding to
8 environment actions.

Start the canonical configuration with:

```bash
conda activate bsp-simple
cd ~/bspline-policy/bspline_policy

HYDRA_FULL_ERROR=1 WANDB_MODE=offline python train.py \
  --config-name=train_drifting_unet_pusht_image_bspline_workspace \
  training.resume=false \
  logging.mode=offline
```

The configuration intentionally preserves the reference Drifting-PushT
settings: `batch_size=64`, `gen_per_label=8`,
`gradient_accumulate_every=1`, per-row loss, temperatures
`[0.02, 0.05, 0.2]`, 300 epochs, and `constant_with_warmup`. Consequently,
each optimizer update evaluates 512 generated trajectories and is intended
for a large-memory training GPU. Inference remains NFE=1.

### Drifting-BSpline Can Image

The Can configuration uses the Robomimic `ph/image.hdf5` demonstrations with
relative 7D actions. It fits cubic B-splines with 16 complete
`[knot, control...]` rows, decodes eight actions for each environment step,
and preserves the faithful Drifting settings (`batch_size=64`,
`gen_per_label=8`, and no gradient accumulation).

The B-spline fitter needs SciPy 1.15, while the existing Robomimic simulator
is installed in `robodiff`. Prepare the replay and train/validation spline
caches once in `bsp-simple`:

```bash
conda activate bsp-simple
cd ~/bspline-policy/bspline_policy

python -m bspline_policy.scripts.prepare_drifting_bspline_can_cache
```

Then train and evaluate in the Robomimic environment:

```bash
conda activate robodiff
cd ~/bspline-policy/bspline_policy

export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/mujoco_py/binaries/linux/mujoco210/bin:${LD_LIBRARY_PATH:-}"

HYDRA_FULL_ERROR=1 WANDB_MODE=offline python train.py \
  --config-name=train_drifting_unet_can_image_bspline_workspace \
  training.resume=false \
  logging.mode=offline
```

The expected source dataset is:

```text
~/bspline-policy/diffusion_policy/data/robomimic/datasets/can/ph/image.hdf5
```

<!-- ## Other Datasets

For the clean Haoyu-left dataset, first convert recorded episodes:

```bash
conda activate bsp-simple
cd ~/bspline-policy/real_env/tidybot2

python convert_clean_bspline_policy_haoyu_left_to_robomimic_real_hdf5.py \
  --input-dir <EPISODE_DIR> \
  --output-path ~/bspline-policy/diffusion_policy/data/clean_bspline_policy_haoyu_left.hdf5 \
  --overwrite
```

Train the clean Haoyu-left B-spline policy:

```bash
conda activate bsp-simple
cd ~/bspline-policy/bspline_policy

HYDRA_FULL_ERROR=1 WANDB_MODE=offline python train.py \
  --config-name=clean_bspline_policy_unet_bspline \
  task=clean_bspline_policy_haoyu_left_bspline \
  task.dataset_path=../diffusion_policy/data/clean_bspline_policy_haoyu_left.hdf5 \
  task.dataset.dataset_path=../diffusion_policy/data/clean_bspline_policy_haoyu_left.hdf5 \
  training.resume=false \
  logging.mode=offline \
  task.dataset.cache_suffix=clean_bspline_policy_haoyu_left_action10_rot6d_v2
```

Train the X5 stack-cube B-spline policy with an existing robomimic HDF5:

```bash
conda activate bsp-simple
cd ~/bspline-policy/bspline_policy

HYDRA_FULL_ERROR=1 WANDB_MODE=offline python train.py \
  --config-name=clean_bspline_policy_unet_bspline \
  task=clean_bspline_policy_stack_cube_teleop_10hz_fix_cam_bspline \
  task.dataset_path=<STACK_CUBE_HDF5> \
  task.dataset.dataset_path=<STACK_CUBE_HDF5> \
  training.resume=false \
  logging.mode=offline \
  task.dataset.cache_suffix=clean_bspline_policy_stack_cube_action20_rot6d_v2
```

After changing action conversion or dataset contents, do not reuse old cache
files. Delete old cache files or use a new `task.dataset.cache_suffix`.

Cache files are created next to the HDF5:

```text
<DATASET_HDF5>.<cache_suffix>.zarr.zip
<DATASET_HDF5>.<cache_suffix>.zarr.zip.lock
<DATASET_HDF5>.<cache_suffix>.bspline_*.pkl
```

## Replay Collected Data

Replay YAM demos with B-spline resampling:

```bash
cd ~/bspline-policy
source real_env/.venv/bin/activate

PYTHONPATH="$PWD/bspline_policy:$PWD/diffusion_policy:${PYTHONPATH:-}" \
python -m bspline_policy.scripts.yam_replay_episodes_bspline \
  --input-dir real_env/yam_teleop/data/demos \
  --speed-up-times 4
```

Replay TidyBot2 demos:

```bash
cd ~/bspline-policy
source real_env/.venv/bin/activate

PYTHONPATH="$PWD/bspline_policy:$PWD/diffusion_policy:$PWD/real_env/tidybot2:${PYTHONPATH:-}" \
python -m bspline_policy.scripts.tidybot2_replay_episodes_bspline \
  --input-dir <EPISODE_DIR> \
  --speed-up-times 4
```

## Policy Rollout

Run a trained B-spline checkpoint through the local rollout wrapper:

```bash
conda activate bsp-simple
cd ~/bspline-policy

PYTHONPATH="$PWD/bspline_policy:$PWD/diffusion_policy:$PWD/real_env/tidybot2:${PYTHONPATH:-}" \
python real_env/tidybot2/rollout_local_policy.py \
  --env tidybot2 \
  --policy bspline \
  --ckpt-path <CKPT_PATH> \
  --diffusion-policy-dir "$PWD/diffusion_policy" \
  --control-freq 200 \
  --data-freq 10 \
  --origin-time-scale 10 \
  --predict-before-end 0.06 \
  --save \
  --output-dir <ROLLOUT_OUTPUT_DIR> \
  --speed-up-times 1.0
```

## Useful Entrypoints

```bash
cd ~/bspline-policy
PYTHONPATH="$PWD/bspline_policy:$PWD/diffusion_policy:$PWD/real_env/tidybot2:${PYTHONPATH:-}" \
python -m bspline_policy.scripts.policy_server_bspline --help
```

```bash
cd ~/bspline-policy
PYTHONPATH="$PWD/bspline_policy:$PWD/diffusion_policy:$PWD/real_env/tidybot2:${PYTHONPATH:-}" \
python -m bspline_policy.scripts.rollout_x5_bspline --help
```

`bspline_policy.scripts.mujoco_bsp_replay` is available for MuJoCo replay, but
it also needs the TidyBot2 MuJoCo control dependencies such as `ruckig`. -->

Next --> [Robot Deployment and Model Inference](../inference/README.md)
