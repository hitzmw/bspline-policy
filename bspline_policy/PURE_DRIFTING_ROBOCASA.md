# Pure Drifting RoboCasa ablation

`train_drifting_unet_turn_off_sink_faucet_image_workspace` trains the
reference one-step Drifting objective directly on dense RoboCasa actions.  It
does not fit, store, predict, project, or decode B-splines.

## Controlled comparison

The pure Drifting and Drifting-BSpline TurnOffSinkFaucet runs share:

- Human-50 demonstrations and the seed-42 train/validation split
- three 224 x 224 RGB observations plus EEF and gripper state
- horizon 16, two observation steps, and eight executed actions
- the same vision encoder, conditional UNet widths, crop, optimizer, EMA,
  learning-rate schedule, 200 epochs, and no accumulation
- Drifting temperatures `[0.02, 0.05, 0.2]`, per-timestep loss, and `G=8`

The pure Drifting run uses the requested batch size 32 (256 generated
trajectories per optimizer step). The completed TurnOffSinkFaucet
Drifting-BSpline run used batch size 8, so this is a known optimization-setting
difference and must be disclosed in the paper table.

The only intended model-side difference is the action representation:

- pure Drifting target: dense physical action trajectory `[16, 7]`
- Drifting-BSpline target: parameter matrix `[16, 8]`, followed by decoding

The direct-action cache is shared with the pure Diffusion baseline.  The `dp`
token in its historical cache suffix is only a filename; the cache contains
ordinary observations and dense 7D actions.

## Train

From `bspline_policy/`:

```bash
conda run -n bsp-simple python train.py \
  --config-name=train_drifting_unet_turn_off_sink_faucet_image_workspace \
  hydra.run.dir=data/outputs/pure_drifting_robocasa/turn_off_sink_faucet_seed42
```

The deterministic output directory and `training.resume=true` allow the same
command to resume `checkpoints/latest.ckpt` after an interruption.

## Evaluate

Use the same Human-50 online protocol as the other ablations: 50 episodes for
each of seeds 195, 196, and 197.  For example:

```bash
conda run -n bsp-robocasa python eval_robocasa.py \
  --checkpoint data/outputs/pure_drifting_robocasa/turn_off_sink_faucet_seed42/checkpoints/epoch=0050-val_loss=VALUE.ckpt \
  --output-dir data/outputs/robocasa_eval_pure_drifting/epoch0050_seed195_n50 \
  --task-name TurnOffSinkFaucet \
  --device cuda:0 \
  --episodes 50 \
  --video-episodes 3 \
  --max-steps 500 \
  --seed 195 \
  --policy-seed 195
```

Repeat with both seed arguments and the output directory changed to 196 and
197.  For the primary controlled table, compare the fixed epoch-50 checkpoint
against the already reported epoch-50 Drifting-BSpline result; report any
post-hoc best-checkpoint comparison separately.
