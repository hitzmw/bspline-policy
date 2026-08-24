"""No-op image runner for real-robot datasets without a simulator.

Training a policy on real-robot data has no sim environment to roll out in;
this runner satisfies the workspace's ``BaseImageRunner`` type check and is
never actually called when ``training.rollout_every`` exceeds the epoch count.
"""

from __future__ import annotations

from typing import Dict

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class NullImageRunner(BaseImageRunner):
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}
