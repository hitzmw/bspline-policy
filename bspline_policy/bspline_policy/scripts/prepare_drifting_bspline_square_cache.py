"""Prepare Square replay and B-spline caches without creating a simulator."""

from __future__ import annotations

from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "config"


def main() -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIRECTORY),
    ):
        config = compose(
            config_name=(
                "train_drifting_unet_square_image_bspline_workspace"
            )
        )
    OmegaConf.resolve(config)

    dataset = hydra.utils.instantiate(config.task.dataset)
    validation_dataset = dataset.get_validation_dataset()
    action_shape = tuple(dataset[0]["action"].shape)
    expected_shape = (
        int(config.horizon),
        int(config.shape_meta.action.shape[0]) + 1,
    )
    if action_shape != expected_shape:
        raise RuntimeError(
            f"Expected cached actions {expected_shape}, got {action_shape}"
        )

    print(f"Square B-spline train samples: {len(dataset)}")
    print(
        "Square B-spline validation samples: "
        f"{len(validation_dataset)}"
    )
    print(f"Square B-spline action shape: {action_shape}")
    print(f"Cache base: {config.task.dataset.cache_base_path}")


if __name__ == "__main__":
    main()
