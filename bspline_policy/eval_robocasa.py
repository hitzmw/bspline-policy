"""Evaluate a trained image-policy checkpoint in RoboCasa v0.2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


# Must be set before importing the checkpoint workspace, which imports modules
# that may transitively import robosuite.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

BSPLINE_POLICY_DIR = Path(__file__).resolve().parent
REPO_ROOT = BSPLINE_POLICY_DIR.parent
DIFFUSION_POLICY_DIR = REPO_ROOT / "diffusion_policy"
for path in (BSPLINE_POLICY_DIR, DIFFUSION_POLICY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import click
import dill
import hydra
import torch
from omegaconf import OmegaConf

from bspline_policy.env_runner.robocasa_image_runner import RoboCasaImageRunner
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


@click.command()
@click.option("-c", "--checkpoint", required=True, type=click.Path(exists=True))
@click.option("-o", "--output-dir", required=True, type=click.Path())
@click.option("-d", "--device", default="cuda:0", show_default=True)
@click.option(
    "--task-name",
    default="TurnOffSinkFaucet",
    show_default=True,
    help="Registered RoboCasa environment name to evaluate.",
)
@click.option("--episodes", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--video-episodes", type=click.IntRange(min=0), default=1, show_default=True)
@click.option("--max-steps", type=click.IntRange(min=1), default=500, show_default=True)
@click.option("--seed", type=int, default=195, show_default=True)
@click.option("--policy-seed", type=int, default=195, show_default=True)
@click.option("--controller-config", type=click.Path(exists=True), default=None)
@click.option("--no-flip-images", is_flag=True)
@click.option("--clip-actions", is_flag=True)
def main(
    checkpoint: str,
    output_dir: str,
    device: str,
    task_name: str,
    episodes: int,
    video_episodes: int,
    max_steps: int,
    seed: int,
    policy_seed: int,
    controller_config: str | None,
    no_flip_images: bool,
    clip_actions: bool,
):
    """Load CHECKPOINT and run online RoboCasa rollouts."""
    output_path = Path(output_dir)
    if output_path.exists():
        raise click.ClickException(f"Output path already exists: {output_path}")
    output_path.mkdir(parents=True)

    with open(checkpoint, "rb") as checkpoint_file:
        payload = torch.load(checkpoint_file, pickle_module=dill)
    cfg = payload["cfg"]
    workspace_class = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = workspace_class(cfg, output_dir=str(output_path))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()

    runner = RoboCasaImageRunner(
        output_dir=str(output_path),
        task_name=task_name,
        n_test=episodes,
        n_test_vis=video_episodes,
        max_steps=max_steps,
        n_obs_steps=int(cfg.n_obs_steps),
        n_action_steps=int(cfg.n_action_steps),
        image_size=224,
        seed=seed,
        policy_seed=policy_seed,
        flip_images=not no_flip_images,
        controller_config_path=controller_config,
        clip_actions=clip_actions,
    )
    metrics = runner.run(policy)
    with (output_path / "eval_log.json").open("w") as output_file:
        json.dump(metrics, output_file, indent=2, sort_keys=True)
    click.echo(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
