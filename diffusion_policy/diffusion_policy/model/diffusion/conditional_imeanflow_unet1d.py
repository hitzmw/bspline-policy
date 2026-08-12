from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D


class ConditionalIMeanFlowUnet1D(nn.Module):
    """Conditional 1D U-Net with improved MeanFlow ``u`` and ``v`` heads.

    Improved MeanFlow conditions its vector field on the interval ``h=t-r``.
    The original image implementation explicitly omits absolute ``t`` from the
    network input.  This wrapper keeps the Diffusion Policy U-Net backbone and
    applies that same interval conditioning.  When ``use_auxiliary_head`` is
    enabled, the final projection produces both average velocity ``u`` and
    instantaneous velocity ``v``.
    """

    def __init__(
            self,
            input_dim: int,
            local_cond_dim: Optional[int] = None,
            global_cond_dim: Optional[int] = None,
            diffusion_step_embed_dim: int = 256,
            down_dims=(256, 512, 1024),
            kernel_size: int = 3,
            n_groups: int = 8,
            cond_predict_scale: bool = False,
            use_auxiliary_head: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.use_auxiliary_head = use_auxiliary_head
        output_dim = input_dim * (2 if use_auxiliary_head else 1)
        self.backbone = ConditionalUnet1D(
            input_dim=input_dim,
            output_dim=output_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            end_timestep: Union[torch.Tensor, float, int],
            local_cond: Optional[torch.Tensor] = None,
            global_cond: Optional[torch.Tensor] = None,
            **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        interval = timestep - end_timestep
        output = self.backbone(
            sample,
            interval,
            local_cond=local_cond,
            global_cond=global_cond,
        )
        if not self.use_auxiliary_head:
            return output
        return output.split(self.input_dim, dim=-1)
