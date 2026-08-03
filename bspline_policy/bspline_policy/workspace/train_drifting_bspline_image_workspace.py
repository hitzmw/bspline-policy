"""Training workspace for one-step Drifting B-spline image policies."""

from __future__ import annotations

import copy
import os
import random

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.workspace.base_workspace import BaseWorkspace


class TrainDriftingBSplineImageWorkspace(BaseWorkspace):
    """Reference Drifting training loop connected to B-spline datasets."""

    include_keys = ("global_step", "epoch")

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = (
            copy.deepcopy(self.model) if cfg.training.use_ema else None
        )
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer,
            params=self.model.parameters(),
        )
        self.global_step = 0
        self.epoch = 0

    @staticmethod
    def _mean_metric_dict(metric_values):
        return {
            key: float(np.mean(values))
            for key, values in metric_values.items()
        }

    @staticmethod
    def _append_metrics(accumulator, metrics):
        for key, value in metrics.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().item()
            accumulator.setdefault(key, []).append(float(value))

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if int(cfg.training.gradient_accumulate_every) != 1:
            raise ValueError(
                "The canonical Drifting-BSpline configuration requires "
                "gradient_accumulate_every=1"
            )

        if cfg.training.resume:
            latest_checkpoint = self.get_checkpoint_path()
            if latest_checkpoint.is_file():
                print(f"Resuming from checkpoint {latest_checkpoint}")
                self.load_checkpoint(path=latest_checkpoint)

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, BaseImageDataset):
            raise TypeError("task.dataset must be a BaseImageDataset")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        validation_dataset = dataset.get_validation_dataset()
        validation_dataloader = DataLoader(
            validation_dataset,
            **cfg.val_dataloader,
        )
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=int(cfg.training.lr_warmup_steps),
            num_training_steps=(
                len(train_dataloader) * int(cfg.training.num_epochs)
            ),
            last_epoch=self.global_step - 1,
        )
        ema: EMAModel | None = None
        if self.ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir,
        )
        if not isinstance(env_runner, BaseImageRunner):
            raise TypeError("task.env_runner must be a BaseImageRunner")

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        train_sampling_batch = None
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            while self.epoch < int(cfg.training.num_epochs):
                self.model.train()
                train_losses = []
                train_metrics = {}
                progress = tqdm.tqdm(
                    train_dataloader,
                    desc=f"Drifting-BSpline epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                )
                for batch_index, batch in enumerate(progress):
                    batch = dict_apply(
                        batch,
                        lambda value: value.to(device, non_blocking=True),
                    )
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    self.optimizer.zero_grad(set_to_none=True)
                    raw_loss, metrics = self.model.compute_loss(
                        batch,
                        return_info=True,
                    )
                    raw_loss.backward()
                    self.optimizer.step()
                    lr_scheduler.step()
                    if ema is not None:
                        ema.step(self.model)

                    loss_value = float(raw_loss.detach().cpu())
                    train_losses.append(loss_value)
                    self._append_metrics(train_metrics, metrics)
                    progress.set_postfix(loss=loss_value, refresh=False)
                    step_log = {
                        "train_loss": loss_value,
                        "global_step": self.global_step,
                        "epoch": self.epoch,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                    step_log.update(
                        {
                            f"train/{key}": values[-1]
                            for key, values in train_metrics.items()
                        }
                    )
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self.global_step += 1

                    max_steps = cfg.training.max_train_steps
                    if max_steps is not None and batch_index + 1 >= max_steps:
                        break
                progress.close()

                epoch_log = {
                    "train_loss": float(np.mean(train_losses)),
                    "global_step": self.global_step,
                    "epoch": self.epoch,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                epoch_log.update(
                    {
                        f"train_epoch/{key}": value
                        for key, value in self._mean_metric_dict(
                            train_metrics
                        ).items()
                    }
                )

                policy = (
                    self.ema_model
                    if self.ema_model is not None
                    else self.model
                )
                policy.eval()

                if self.epoch % int(cfg.training.rollout_every) == 0:
                    epoch_log.update(env_runner.run(policy))

                if self.epoch % int(cfg.training.val_every) == 0:
                    validation_losses = []
                    validation_metrics = {}
                    with torch.no_grad():
                        validation_progress = tqdm.tqdm(
                            validation_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        )
                        for batch_index, batch in enumerate(
                            validation_progress
                        ):
                            batch = dict_apply(
                                batch,
                                lambda value: value.to(
                                    device,
                                    non_blocking=True,
                                ),
                            )
                            loss, metrics = policy.compute_loss(
                                batch,
                                return_info=True,
                            )
                            validation_losses.append(
                                float(loss.detach().cpu())
                            )
                            self._append_metrics(
                                validation_metrics,
                                metrics,
                            )
                            max_steps = cfg.training.max_val_steps
                            if (
                                max_steps is not None
                                and batch_index + 1 >= max_steps
                            ):
                                break
                        validation_progress.close()
                    if validation_losses:
                        epoch_log["val_loss"] = float(
                            np.mean(validation_losses)
                        )
                        epoch_log.update(
                            {
                                f"val/{key}": value
                                for key, value in self._mean_metric_dict(
                                    validation_metrics
                                ).items()
                            }
                        )

                if (
                    train_sampling_batch is not None
                    and self.epoch % int(cfg.training.sample_every) == 0
                ):
                    with torch.no_grad():
                        prediction = policy.predict_action(
                            train_sampling_batch["obs"]
                        )["action_pred"]
                        target = train_sampling_batch["action"]
                        epoch_log["train_action_mse_error"] = float(
                            torch.nn.functional.mse_loss(
                                prediction,
                                target,
                            )
                            .detach()
                            .cpu()
                        )

                if self.epoch % int(cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    checkpoint_metrics = {
                        key.replace("/", "_"): value
                        for key, value in epoch_log.items()
                    }
                    topk_path = topk_manager.get_ckpt_path(
                        checkpoint_metrics
                    )
                    if topk_path is not None:
                        self.save_checkpoint(path=topk_path)

                wandb_run.log(epoch_log, step=self.global_step)
                json_logger.log(epoch_log)
                self.epoch += 1

        wandb_run.finish()
