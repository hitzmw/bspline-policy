"""Offline runner used when an environment is not part of the checkout."""

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class OfflineImageRunner(BaseImageRunner):
    """Keep training operational while making missing rollouts explicit."""

    def run(self, policy):
        del policy
        return {"offline_rollout": 1.0}
