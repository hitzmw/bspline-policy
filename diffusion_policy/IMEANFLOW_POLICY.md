# iMeanFlow Policy

This implementation replaces the DDPM objective and scheduler in the original
Diffusion Policy U-Net with improved MeanFlow while keeping the dataset,
normalizer, observation encoder, policy API, workspace, EMA, rollout, and
checkpoint code unchanged.

The port follows the two local references:

- `../imeanflow`: logit-normal `(t, r)` sampling, the improved v-loss,
  auxiliary velocity head, adaptive weighting, and interval-conditioned model.
- `../dmpo-release`: the PyTorch trajectory JVP and action-flow update.

The training path uses

`z_t = (1-t) x + t e`, `v = e-x`, and
`V = u + (t-r) stopgrad(du/dt)`.

The JVP tangent uses the predicted boundary velocity. The inference path uses
`z_r = z_t - (t-r) u`; therefore the default `num_inference_steps: 1` performs
one U-Net evaluation from noise at `t=1` to an action trajectory at `r=0`.

Low-dimensional training:

```bash
python train.py --config-name=train_imeanflow_unet_lowdim_workspace
```

Image-conditioned training:

```bash
python train.py --config-name=train_imeanflow_unet_hybrid_workspace
```

Important policy settings are `data_proportion`, `p_mean`, `p_std`,
`adaptive_loss`, `norm_p`, `norm_eps`, `auxiliary_loss_weight`, and
`num_inference_steps`. Increasing `num_inference_steps` enables the same
interval update over a uniform multi-step schedule without retraining.
