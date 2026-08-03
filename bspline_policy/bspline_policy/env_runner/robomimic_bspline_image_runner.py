"""Robomimic runner compatibility adapter for the Can B-spline task."""

from __future__ import annotations

import inspect

from diffusion_policy.env_runner.robomimic_image_runner import (
    RobomimicImageRunner,
)
from diffusion_policy.gym_util import async_vector_env


def _patch_gym_vector_utility_order() -> None:
    """Bridge the Gym 0.21 and newer vector-utility call signatures.

    The repository's vector environment uses the newer argument order, while
    the canonical ``robodiff`` environment contains Gym 0.21. Patch only the
    three imported utility functions when their signatures identify the old
    API. The patch happens before worker processes are forked.
    """
    read_parameters = list(
        inspect.signature(
            async_vector_env.read_from_shared_memory
        ).parameters
    )
    if read_parameters[:2] == ["shared_memory", "space"]:
        original_read = async_vector_env.read_from_shared_memory

        def read_from_shared_memory(space, shared_memory, n=1):
            return original_read(shared_memory, space, n=n)

        async_vector_env.read_from_shared_memory = read_from_shared_memory

    write_parameters = list(
        inspect.signature(
            async_vector_env.write_to_shared_memory
        ).parameters
    )
    if write_parameters[:2] == ["index", "value"]:
        original_write = async_vector_env.write_to_shared_memory

        def write_to_shared_memory(space, index, value, shared_memory):
            return original_write(index, value, shared_memory, space)

        async_vector_env.write_to_shared_memory = write_to_shared_memory

    concatenate_parameters = list(
        inspect.signature(async_vector_env.concatenate).parameters
    )
    if concatenate_parameters[:2] == ["items", "out"]:
        original_concatenate = async_vector_env.concatenate

        def concatenate(space, items, out):
            return original_concatenate(items, out, space)

        async_vector_env.concatenate = concatenate


class RobomimicBSplineImageRunner(RobomimicImageRunner):
    """Run decoded B-spline actions under either supported Gym API."""

    def __init__(self, *args, **kwargs):
        _patch_gym_vector_utility_order()
        super().__init__(*args, **kwargs)
