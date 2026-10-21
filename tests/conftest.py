import pytest
import torch

from evolution.core import config


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@pytest.fixture(autouse=True)
def _no_grad_and_seed():
    """The simulation always runs with autograd off (see launch.py). Seed for repeatability where
    possible (some kernels use atomics, so runs are not bit-for-bit deterministic)."""
    torch.manual_seed(0)
    with torch.no_grad():
        yield


def small_cfg(**overrides) -> config.Config:
    """A small world that builds in well under a second."""
    kwargs = dict(start_creatures=512, max_creatures=4096, size=150, init_food_scale=12.0)
    kwargs.update(overrides)
    return config.Config(**kwargs)


@pytest.fixture
def make_world():
    """Factory for small GWorlds; each test gets fresh ones."""
    from evolution.core.gworld import GWorld

    def _make(**overrides):
        return GWorld(small_cfg(**overrides))
    return _make
