"""Evaluate a B-spline policy checkpoint with optional runner overrides."""

from __future__ import annotations

import json
import os
import pathlib
import sys


BSPLINE_POLICY_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = BSPLINE_POLICY_DIR.parent
DIFFUSION_POLICY_DIR = REPO_ROOT / "diffusion_policy"
for path in (BSPLINE_POLICY_DIR, DIFFUSION_POLICY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import click
import dill
import hydra
import torch
import wandb
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace


sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command()
@click.option("-c", "--checkpoint", required=True, type=click.Path(exists=True))
@click.option("-o", "--output-dir", required=True, type=click.Path())
@click.option("-d", "--device", default="cuda:0", show_default=True)
@click.option("--n-envs", type=click.IntRange(min=1), default=None)
def main(checkpoint: str, output_dir: str, device: str, n_envs: int | None):
    """Load CHECKPOINT and write rollout metrics under OUTPUT_DIR."""
    output_path = pathlib.Path(output_dir)
    if output_path.exists():
        raise click.ClickException(f"Output path already exists: {output_path}")
    output_path.mkdir(parents=True)

    with open(checkpoint, "rb") as checkpoint_file:
        payload = torch.load(checkpoint_file, pickle_module=dill)
    cfg = payload["cfg"]
    if n_envs is not None:
        cfg.task.env_runner.n_envs = n_envs

    workspace_class = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = workspace_class(cfg, output_dir=str(output_path))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=str(output_path),
    )
    runner_log = env_runner.run(policy)

    json_log = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = float(value)
    with output_path.joinpath("eval_log.json").open("w") as output_file:
        json.dump(json_log, output_file, indent=2, sort_keys=True)

    click.echo(f"test_mean_score={json_log.get('test/mean_score')}")


if __name__ == "__main__":
    main()
