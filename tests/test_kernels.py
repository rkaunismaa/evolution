"""CUDA kernels checked against plain-PyTorch reference implementations on small random inputs."""
import pytest
import torch

from tests.conftest import requires_cuda

pytestmark = requires_cuda


def _reference_eat_grow(world):
    """Pure-PyTorch version of Creatures.eat_grow (kernels setup_eat_grid, eat, grow).
    Returns new (food_grid, energies, healths, ages, age_mults)."""
    cfg, c = world.cfg, world.creatures
    assert cfg.alive_cost.expr == '0.01f * x', "reference hard-codes the default alive_cost"
    food_grid = world.food_grid.clone()
    W = food_grid.shape[0]
    pos = c.positions.data.int() + c.pad
    cell = (pos[:, 1] * W + pos[:, 0]).long()
    eat_pct, sizes = c.eat_pcts.data, c.sizes.data
    energies, healths = c.energies.data.clone(), c.healths.data.clone()
    ages, age_mults = c.ages.data.clone(), c.age_mults.data.clone()

    # setup_eat_grid: total share claimed in each cell
    pct_eaten = torch.zeros(W * W, device=food_grid.device).index_add_(0, cell, eat_pct)
    claimed = pct_eaten[cell]

    # eat
    food = food_grid.view(-1)[cell]
    food = torch.where(claimed > 1, food * eat_pct / claimed, food * eat_pct)
    old = ages > cfg.age_old_mul * c.age_speeds.data
    age_mults = torch.where(old, age_mults * (1 + cfg.age_dmg_mul), age_mults)
    neg = food < 0
    healths = healths + torch.where(neg, food * cfg.neg_food_eat_mul * age_mults,
                                    food * cfg.food_health_recovery)
    food = torch.where(neg, torch.zeros_like(food), food)
    alive_cost = 0.01 * sizes * age_mults
    energies = energies + food / age_mults - alive_cost
    # note: the kernel removes a flat food_cover_decr per creature (its per-creature
    # creature_food_decr is computed but unused); the reference mirrors the kernel
    updates = torch.zeros(W * W, device=food_grid.device).index_add_(0, cell, food + cfg.food_cover_decr)
    ages = ages + 1

    # grow (padding untouched)
    f = food_grid.view(-1) - updates
    growth = cfg.food_step_size * (cfg.max_food - f)
    f = torch.where(f < 0, f + growth * cfg.food_recovery_rate,
                    torch.where(f > cfg.max_food, f + growth * cfg.food_decay_rate,
                                f + growth * cfg.food_growth_rate))
    grown = f.view(W, W)
    p = c.pad
    new_grid = food_grid.clone()
    new_grid[p:W - p, p:W - p] = grown[p:W - p, p:W - p]
    return new_grid, energies, healths, ages, age_mults


@pytest.mark.parametrize("crowded", [False, True])
def test_eat_grow_matches_reference(make_world, crowded):
    # crowded: many creatures on a few cells, so cells are over-claimed (pct_eaten > 1)
    world = make_world(size=40) if crowded else make_world()
    c = world.creatures
    for _ in range(30):     # let ages, energies and food levels spread out
        world.step()
    if crowded:
        c.positions.data[:] = torch.rand_like(c.positions.data) * 3 + 10
    world.food_grid[world.food_grid.shape[0] // 2] = -1.0   # a row of negative food
    c.ages.data[: c.population // 4] = 1e6                  # some old creatures

    ref = _reference_eat_grow(world)
    c.eat_grow(world.food_grid, world.state)
    got = (world.food_grid, c.energies.data, c.healths.data, c.ages.data, c.age_mults.data)
    for name, g, r in zip(["food_grid", "energies", "healths", "ages", "age_mults"], got, ref):
        torch.testing.assert_close(g, r, rtol=1e-4, atol=1e-4, msg=name)


def test_eat_scratch_buffers_are_rezeroed(make_world):
    world = make_world()
    world.step()   # the first step only computes decisions; eating starts on the second
    for _ in range(20):
        world.step()
        pct_eaten, food_grid_updates = world.creatures._eat_scratch
        assert pct_eaten.count_nonzero().item() == 0
        assert food_grid_updates.count_nonzero().item() == 0


def test_hsv_spiral_maps_zero_to_black_and_colors_into_range(make_world):
    from evolution.core.creatures.trait_initializer import cuda_hsv_spiral
    world = make_world()
    colors = torch.linspace(0, 255, 256, device='cuda').unsqueeze(1)
    rgb = cuda_hsv_spiral(colors, world.kernels)
    assert rgb.shape == (256, 3)
    assert torch.equal(rgb[0], torch.zeros(3, device='cuda'))
    assert rgb.min() >= 0 and rgb.max() <= 255
