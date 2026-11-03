"""Species-specific food (Config.food_types > 1): each colour band eats, sees and drops its own food."""
import pytest
import torch

from tests.conftest import requires_cuda

pytestmark = requires_cuda


def test_colour_bands_choose_food_type(make_world):
    world = make_world(food_types=3)
    c = world.creatures
    c.colors.data[:6] = torch.tensor([1.0, 85.0, 85.7, 170.0, 170.4, 255.0], device='cuda')
    assert c.food_types()[:6].tolist() == [0, 0, 1, 1, 2, 2]
    assert c.food_types().dtype == torch.int32
    assert make_world().creatures.food_types().eq(0).all()     # one food: everyone eats type 0


def test_grid_shapes(make_world):
    world = make_world(food_types=3, food_capacity_variation=0.5, food_capacity_scale=25.0)
    side = world.cfg.size + 2 * world.creatures.pad
    assert world.food_grid.shape == (3, side, side)
    assert world.food_capacity.shape == (3, side, side)
    assert world.central_food_grid.shape == (3, world.cfg.size, world.cfg.size)


def test_creatures_see_only_their_own_food(make_world):
    world = make_world(food_types=3)
    c = world.creatures
    world.step()
    for t in range(3):     # make the types easy to tell apart
        world.food_grid[t] = float(t + 1) * 10
    stimuli = c.collect_stimuli(world.collisions, world.food_grid)
    start = c.cfg.num_rays + c.cfg.mem_size
    food_inputs = stimuli[:, start:start + c.num_food]
    expected = (c.food_types().float() + 1) * 10
    assert torch.equal(food_inputs, expected[:, None].expand_as(food_inputs))


def test_eating_only_takes_own_food(make_world):
    world = make_world(food_types=2, food_cover_decr=0.0, food_grow_every=1000)
    c = world.creatures
    world.step()
    c.colors.data[:] = 10.0                       # everyone eats type 0
    before = world.food_grid.clone()
    world.state.time += 1
    c.eat_grow(world.food_grid, world.state, world.food_capacity)
    assert (world.food_grid[0] < before[0]).any()
    assert torch.equal(world.food_grid[1], before[1])


def test_dead_creatures_drop_their_own_food(make_world):
    world = make_world(food_types=3)
    c = world.creatures
    world.step()
    c.colors.data[:] = 200.0                      # type 2
    c.energies.data[:10] = -1.0                   # ten deaths
    before = world.food_grid.clone()
    c.fused_kill_reproduce(world.central_food_grid, world.state)
    gained = world.food_grid - before
    assert gained[2].sum() > 0
    assert gained[0].abs().sum() == 0 and gained[1].abs().sum() == 0


def test_each_type_has_its_own_rich_land(make_world):
    world = make_world(food_types=3, food_capacity_variation=0.5, food_capacity_scale=25.0)
    p = world.creatures.pad
    central = world.food_capacity[:, p:-p, p:-p].flatten(1)
    for t in range(3):
        assert abs(central[t].std().item() - 0.375 * 0.5) < 0.01   # same contrast for every type
    corr = torch.corrcoef(central)
    assert corr[0, 1].abs() < 0.3 and corr[0, 2].abs() < 0.3 and corr[1, 2].abs() < 0.3


def test_invariants_with_three_foods(make_world):
    # full food per type: this small world goes extinct when three types share one type's food
    world = make_world(food_types=3, food_types_share=0, food_capacity_variation=0.5, food_capacity_scale=25.0,
                       kill_reproduce_every=4, brain_dtype='float16', food_grow_every=4,
                       food_capacity_change_steps=300, food_capacity_update_every=10)
    p = world.creatures.pad
    for step in range(200):
        assert world.step(), f"mass extinction at step {step}"
    assert torch.isfinite(world.food_grid).all()
    assert world.food_grid[:, :p].eq(0).all() and world.food_grid[:, -p:].eq(0).all()
    assert world.food_grid[:, :, :p].eq(0).all() and world.food_grid[:, :, -p:].eq(0).all()


@pytest.mark.parametrize("quantized", [False, True])
def test_checkpoint_round_trip_with_three_foods(make_world, tmp_path, quantized):
    from evolution.core.gworld import GWorld
    world = make_world(food_types=3, food_capacity_variation=0.5, food_capacity_scale=25.0,
                       food_capacity_change_steps=500)
    for _ in range(30):
        world.step()
    path = tmp_path / "foods.ckpt"
    world.write_checkpoint(str(path), quantized=quantized)
    loaded, _ = GWorld.from_checkpoint(str(path))
    assert loaded.food_grid.shape == world.food_grid.shape and loaded.food_grid.is_contiguous()
    if quantized:   # 8-bit per column: within one step of each column's range
        span = (world.food_grid.amax((0, 1)) - world.food_grid.amin((0, 1)).clamp(max=0))
        assert ((loaded.food_grid - world.food_grid).abs() <= span / 255 + 1e-5).all()
    else:
        assert torch.equal(loaded.food_grid, world.food_grid)
    world.food_capacity_field.update(world.state.time)
    torch.testing.assert_close(loaded.food_capacity, world.food_capacity, rtol=0, atol=1e-6)
    loaded.step()


def test_older_capacity_field_state_loads(make_world):
    """Saves from before food_types stored [modes] wave parameters and a float norm."""
    world = make_world(food_capacity_variation=0.5, food_capacity_scale=25.0)
    field = world.food_capacity_field
    expected = field.multiplier.clone()
    old = {k: (v[0] if isinstance(v, torch.Tensor) and v.dim() == 2 else v) for k, v in field.state_dict().items()}
    old['norm'] = float(field.norm[0])
    field.load_state_dict(old, 0)
    torch.testing.assert_close(field.multiplier, expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("scaled", [0, 1], ids=["flat-cost", "scaled-cost"])
def test_cell_panel_totals_over_food_types(make_world, scaled):
    world = make_world(size=40, food_types=3, food_cover_scaled=scaled, food_cover_decr=0.2)
    c, cfg = world.creatures, world.cfg
    for _ in range(5):
        world.step()
    pos = c.positions.data.int()
    cells, counts = torch.unique(pos, dim=0, return_counts=True)
    x, y = cells[counts.argmax()].tolist()
    world.state.selected_cell.set_cell((x, y))
    in_cell = (pos[:, 0] == x) & (pos[:, 1] == y)
    eat_pcts = c.eat_pcts.data[in_cell].clone()
    world.state.time += 1
    c.eat_grow(world.food_grid, world.state, world.food_capacity)
    panel = world.state.selected_cell
    if scaled:
        expected = (cfg.food_cover_decr * cfg.food_cover_decr_pct * eat_pcts).sum().item()
    else:
        expected = cfg.food_cover_decr * in_cell.sum().item()
    assert panel.cover_cost == pytest.approx(expected, rel=1e-4)
    assert len(panel.foods) == 3
    assert panel.food == pytest.approx(sum(panel.foods))


@pytest.mark.parametrize("share", [1, 0], ids=["shared", "full"])
def test_food_types_share_the_worlds_food(make_world, share):
    """Shared (the default): each type grows towards max_food / food_types, so total food matches one
    type. Unshared: every type grows towards the full max_food."""
    world = make_world(food_types=3, food_types_share=share, food_step_size=0.02, food_cover_decr=0.0)
    c, p = world.creatures, world.creatures.pad
    per_type = world.cfg.max_food / 3 if share else world.cfg.max_food
    assert world.cfg.max_food_per_type() == pytest.approx(per_type)
    assert world.food_grid.max() <= world.cfg.init_food_scale * (1 / 3 if share else 1)   # initial food too
    c.positions.data[:] = 1.0              # park everyone in one corner cell
    c.eat_pcts.data[:] = 0.0               # and stop them eating
    for _ in range(60):
        world.state.time += 1
        c.eat_grow(world.food_grid, world.state, world.food_capacity)
    food = world.food_grid[:, p + 5:-p, p + 5:-p]
    assert food.mean().item() == pytest.approx(per_type, rel=0.02)


def test_one_food_type_is_unaffected_by_sharing(make_world):
    assert make_world(food_types=1, food_types_share=1).cfg.max_food_per_type() == 15.0


def test_older_configs_share_by_default():
    """Configs pickled before food_types_share existed fall back to the class default (shared)."""
    import pickle
    from evolution.core.config import Config
    cfg = Config(food_types=3)
    del cfg.__dict__['food_types_share']
    old = pickle.loads(pickle.dumps(cfg))
    assert old.max_food_per_type() == pytest.approx(5.0)
    del cfg.__dict__['food_types']      # and from before food_types: one full food
    assert pickle.loads(pickle.dumps(cfg)).max_food_per_type() == 15.0
