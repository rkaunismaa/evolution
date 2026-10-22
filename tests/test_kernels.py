"""CUDA kernels checked against plain-PyTorch reference implementations on small random inputs."""
import pytest
import torch

from tests.conftest import requires_cuda

pytestmark = requires_cuda


def _reference_eat_grow(world, grow=True, step_mult=1, capacity=None):
    """Pure-PyTorch version of Creatures.eat_grow (kernels setup_eat_grid, eat, grow).
    grow=False applies only the eating updates; step_mult scales the growth step.
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
    if cfg.food_cover_scaled:
        cover = cfg.food_cover_decr * eat_pct * cfg.food_cover_decr_pct
    else:
        cover = torch.full_like(food, cfg.food_cover_decr)
    updates = torch.zeros(W * W, device=food_grid.device).index_add_(0, cell, food + cover)
    ages = ages + 1

    if not grow:
        return (food_grid - updates.view(W, W)), energies, healths, ages, age_mults

    # grow (padding untouched)
    f = food_grid.view(-1) - updates
    cap = cfg.max_food if capacity is None else cfg.max_food * capacity.view(-1)
    growth = cfg.food_step_size * step_mult * (cap - f)
    f = torch.where(f < 0, f + growth * cfg.food_recovery_rate,
                    torch.where(f > cap, f + growth * cfg.food_decay_rate,
                                f + growth * cfg.food_growth_rate))
    grown = f.view(W, W)
    p = c.pad
    new_grid = food_grid.clone()
    new_grid[p:W - p, p:W - p] = grown[p:W - p, p:W - p]
    return new_grid, energies, healths, ages, age_mults


@pytest.mark.parametrize("scaled", [0, 1], ids=["flat-cover", "scaled-cover"])
@pytest.mark.parametrize("crowded", [False, True])
def test_eat_grow_matches_reference(make_world, crowded, scaled):
    # crowded: many creatures on a few cells, so cells are over-claimed (pct_eaten > 1)
    world = make_world(size=40, food_cover_scaled=scaled) if crowded else make_world(food_cover_scaled=scaled)
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


@pytest.mark.parametrize("crowded", [False, True])
def test_batched_food_growth_matches_reference(make_world, crowded):
    """food_grow_every=4: steps that are not multiples of 4 only apply eating (at creature cells);
    multiples of 4 grow the whole grid by 4x the step."""
    world = make_world(size=40, food_grow_every=4) if crowded else make_world(food_grow_every=4)
    c = world.creatures
    for _ in range(30):
        world.step()
    if crowded:
        c.positions.data[:] = torch.rand_like(c.positions.data) * 3 + 10
    names = ["food_grid", "energies", "healths", "ages", "age_mults"]
    for t, grows in ((41, False), (42, False), (44, True)):
        world.state.time = t
        ref = _reference_eat_grow(world, grow=grows, step_mult=4)
        c.eat_grow(world.food_grid, world.state)
        got = (world.food_grid, c.energies.data, c.healths.data, c.ages.data, c.age_mults.data)
        for name, g, r in zip(names, got, ref):
            torch.testing.assert_close(g, r, rtol=1e-4, atol=1e-4, msg=f"{name} at time {t}")


def test_grow_with_capacity_map_matches_reference(make_world):
    world = make_world(food_capacity_variation=0.6, food_capacity_scale=30.0)
    c = world.creatures
    for _ in range(30):
        world.step()
    # push some cells above their local capacity so the decay branch runs too
    world.food_grid[c.pad:-c.pad, c.pad:c.pad + 20] = 30.0
    ref = _reference_eat_grow(world, capacity=world.food_capacity)
    c.eat_grow(world.food_grid, world.state, world.food_capacity)
    torch.testing.assert_close(world.food_grid, ref[0], rtol=1e-4, atol=1e-4)
