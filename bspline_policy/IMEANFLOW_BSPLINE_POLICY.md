# iMeanFlow B-spline Policy

This implementation keeps the existing offline adaptive B-spline fitting and
replaces DDPM with the improved MeanFlow objective. Training combines the
iMeanFlow parameter-space loss with a differentiable one-step curve
reconstruction loss:

```text
raw action -> fitted [U*, C*] -> normalized x0
                              |-> random (t,r) improved MeanFlow loss
                              `-> same noise, exact 1->0 map, hard clip
                                  -> STE knot projection
                                  -> PyTorch B-spline decode
                                  -> normalized physical-action MSE
```

The reconstruction coefficient defaults to `0.05` and is linearly warmed from
zero over the first 10% of optimizer steps. Set
`policy.reconstruction_loss_weight=0` for the parameter-loss-only ablation.

## Decoder contract

- Relative knots are intentionally unsupported in the first version and fail
  fast unless `relative_knots: false`.
- The effective cubic support is `[U[3], U[-4]]`.
- The default `raw_time_clamp` mode evaluates at physical local timestamps
  `j = 0, ..., n_action_steps-1` and clamps each timestamp to the effective
  support. This endpoint-hold rule is identical in training and rollout and
  cannot emit SciPy-style extrapolation NaNs.
- Rollout and reconstruction both use the same PyTorch de Boor decoder.
- `support_linspace` remains available as an eval-only decoding ablation. To
  reproduce the old rollout convention as closely as possible, override both
  `policy.rollout_decode_mode=support_linspace` and
  `policy.min_knot_delta=1e-6`.
- The generated normalized parameters use the same hard `[-1,1]` clamp in the
  reconstruction branch and rollout.

The DDPM and Drifting policies keep their previous rollout decoders, so a
success-rate comparison against those baselines changes both the generative
loss and the decoding-time convention. Use the `support_linspace` eval-only
ablation when separating those variables in reporting.

## Configurations

From `bspline_policy/`:

```bash
# Push-T
python train.py \
  --config-name=train_imeanflow_unet_pusht_image_bspline_workspace

# Robomimic Square
python train.py \
  --config-name=train_imeanflow_unet_square_image_bspline_workspace

# Robomimic Can uses the same policy/workspace with the Can task group
python train.py \
  --config-name=train_imeanflow_unet_square_image_bspline_workspace \
  task=can_image_bspline \
  name=train_imeanflow_unet_can_image_bspline
```

These commands are launch examples only; local validation did not run a
training job.

## Logged diagnostics

The workspace logs total, iMeanFlow `u`/`v`, and reconstruction losses plus:

- `error_fit`: fitted target curve vs raw action;
- `error_parameter_to_curve`: generated curve vs fitted target curve;
- `error_final`: generated curve vs raw action;
- `clip_fraction` and `knot_violation_fraction`;
- generated and target support coverage;
- the current warmup-adjusted reconstruction weight.

All three curve errors use `normalizer["raw_action"]` and the same valid mask,
so they are comparable in one normalized physical-action space. Episode-tail
targets are zero padded and masked through a dedicated sampler path; they do
not pass through the generic NaN-padding assertion.
