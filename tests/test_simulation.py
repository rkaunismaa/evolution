"""Whole-simulation checks: invariants over many steps, config options, checkpoint round trips."""
import pytest
import torch

from tests.conftest import requires_cuda

pytestmark = requires_cuda


def _check_invariants(world):
    c, cfg = world.creatures, world.cfg
    assert 0 < c.population <= cfg.max_creatures
    lo, hi = c.posn_bounds
    assert c.positions.data.min() >= lo and c.positions.data.max() <= hi
    for name, v in c.traits.variables.items():
        assert torch.isfinite(v.data).all(), name
    assert torch.isfinite(world.food_grid).all()
    p = c.pad   # padding around the food grid never changes
    assert world.food_grid[:p].eq(0).all() and world.food_grid[-p:].eq(0).all()
    assert world.food_grid[:, :p].eq(0).all() and world.food_grid[:, -p:].eq(0).all()


@pytest.mark.parametrize("opts", [
    {},
    {"kill_reproduce_every": 4},
    {"brain_dtype": "float16"},
    {"kill_reproduce_every": 4, "brain_dtype": "float16"},
    {"kill_reproduce_every": 4, "brain_dtype": "float16", "food_grow_every": 4},
], ids=["default", "every4", "fp16", "every4+fp16", "large-preset-options"])
def test_invariants_hold_over_many_steps(make_world, opts):
    world = make_world(**opts)
    for step in range(200):
        assert world.step(), f"mass extinction at step {step}"
        if step % 20 == 0:
            _check_invariants(world)
    _check_invariants(world)
    assert world.state.time == 200


def test_kill_reproduce_every_only_runs_on_multiples(make_world, monkeypatch):
    world = make_world(kill_reproduce_every=3)
    calls = []
    real = world.creatures.fused_kill_reproduce
    monkeypatch.setattr(world.creatures, 'fused_kill_reproduce',
                        lambda *a, **k: (calls.append(world.state.time), real(*a, **k)))
    for _ in range(12):
        world.step()
    # step() skips update_creatures on the very first call, before any outputs exist
    assert calls == [3, 6, 9]


def test_float16_brains_are_close_to_float32(make_world):
    world = make_world()
    world.step()
    c = world.creatures
    stimuli = c.collect_stimuli(world.collisions, world.food_grid).unsqueeze(1)
    mem = c.memories.data.clone()
    out32 = c.forward(stimuli)
    c.memories.data[:] = mem
    for w in c.weights:
        w.dtype = torch.float16
        w._data = w._data.half()
        w.data = w._data[c.traits.start_idx:c.traits.start_idx + c.population]
    out16 = c.forward(stimuli)
    assert out16.dtype == torch.float32
    assert (out32 - out16).abs().mean().item() < 2e-3


def test_float16_storage_and_children(make_world):
    world = make_world(brain_dtype="float16", start_creatures=1024)
    c = world.creatures
    assert all(w._data.dtype == torch.float16 for w in c.weights)
    assert all(b._data.dtype == torch.float32 for b in c.biases)
    world.step()
    c.ages.data[:] = 1000.        # everyone mature and well fed, so many reproduce at once
    c.energies.data[:] = 1000.
    for _ in range(5):
        world.step()
    assert c.n_children.data.sum() > 0, "no reproduction happened"
    assert all(w._data.dtype == torch.float16 for w in c.weights)


@pytest.mark.parametrize("brain_dtype", ["float32", "float16"])
def test_checkpoint_round_trip_unquantized_is_exact(make_world, tmp_path, brain_dtype):
    from evolution.core.gworld import GWorld
    world = make_world(brain_dtype=brain_dtype)
    for _ in range(50):
        world.step()
    path = tmp_path / "world.ckpt"
    world.write_checkpoint(str(path), quantized=False)
    loaded, _ = GWorld.from_checkpoint(str(path))

    assert loaded.state.time == world.state.time
    assert repr(loaded.cfg) == repr(world.cfg)   # function fields compare by identity, so use repr
    assert torch.equal(loaded.food_grid, world.food_grid)
    assert loaded.creatures.population == world.creatures.population
    for name, v in world.creatures.traits.variables.items():
        got = loaded.creatures.traits.variables[name].data
        assert got.dtype == v.data.dtype, name
        assert torch.equal(got, v.data), name


def test_checkpoint_round_trip_quantized_is_close(make_world, tmp_path):
    from evolution.core.gworld import GWorld
    world = make_world()
    for _ in range(50):
        world.step()
    path = tmp_path / "world.ckpt"
    world.write_checkpoint(str(path), quantized=True)
    loaded, _ = GWorld.from_checkpoint(str(path))
    exact = ['positions', 'colors', 'energies', 'healths', 'ages', 'n_children']
    for name, v in world.creatures.traits.variables.items():
        got = loaded.creatures.traits.variables[name].data
        if name in exact:
            assert torch.equal(got, v.data), name
        else:   # 8-bit per-column quantization over a range stretched to include 0:
                # error within one step of that range (steps are at least SMALL_SCALE_THRESHOLD)
            d = v.data.float()
            span = d.amax(0).clamp(min=0) - d.amin(0).clamp(max=0)
            step = (span / 255).clamp(min=6.1e-5)
            assert ((got.float() - d).abs() <= step + 1e-6).all(), name
    loaded.step()   # and the loaded world keeps running


def test_food_capacity_map_off_by_default(make_world):
    assert make_world().food_capacity is None


def test_food_capacity_map_shape_range_and_smoothness(make_world):
    world = make_world(food_capacity_variation=0.5, food_capacity_scale=25.0)
    cap = world.food_capacity
    assert cap.shape == world.food_grid.shape and cap.is_contiguous()
    p = world.creatures.pad
    central = cap[p:-p, p:-p]
    assert central.min() >= 0.0
    assert abs(central.std().item() - 0.375 * 0.5) < 0.01       # contrast set by the variation
    inside = ((central >= 0.5) & (central <= 1.5)).float().mean().item()
    assert inside > 0.97                                        # nearly all land within 1 +- variation
    assert abs(central.mean().item() - 1.0) < 1e-3  # waves average out exactly over the world
    assert cap[:p].eq(1).all()                      # padding is neutral
    # smooth: neighbouring cells differ far less than the overall spread
    spread = (central.max() - central.min()).item()
    assert (central[1:] - central[:-1]).abs().max() < spread / 4
    assert (central[:, 1:] - central[:, :-1]).abs().max() < spread / 4
    # tiles across the world's edges: the wrap-around step is as smooth as any other
    assert (central[0] - central[-1]).abs().max() < spread / 4


def test_food_settles_near_local_capacity(make_world):
    """With no creatures eating, food converges towards max_food * capacity in each cell."""
    world = make_world(food_capacity_variation=0.5, food_capacity_scale=25.0, food_step_size=0.02,
                       food_cover_decr=0.0, init_food_scale=0.0)
    c, p = world.creatures, world.creatures.pad
    c.positions.data[:] = 1.0              # park everyone in one corner cell
    c.eat_pcts.data[:] = 0.0               # and stop them eating
    for _ in range(40):
        world.state.time += 1
        c.eat_grow(world.food_grid, world.state, world.food_capacity)
    food = world.food_grid[p:-p, p:-p]
    target = world.cfg.max_food * world.food_capacity[p:-p, p:-p]
    corr = torch.corrcoef(torch.stack([food.flatten(), target.flatten()]))[0, 1]
    assert corr > 0.99


@pytest.mark.parametrize("change_steps", [0, 500], ids=["static", "drifting"])
def test_food_capacity_field_saved_in_checkpoint(make_world, tmp_path, change_steps):
    from evolution.core.gworld import GWorld
    world = make_world(food_capacity_variation=0.5, food_capacity_scale=25.0,
                       food_capacity_change_steps=change_steps, food_capacity_update_every=10)
    for _ in range(25):
        world.step()
    path = tmp_path / "cap.ckpt"
    world.write_checkpoint(str(path))
    loaded, _ = GWorld.from_checkpoint(str(path))
    # the saved field is recomputed for the saved time: same as the running world's latest update
    world.food_capacity_field.update(world.state.time)
    torch.testing.assert_close(loaded.food_capacity, world.food_capacity, rtol=0, atol=1e-6)
    # and both keep evolving identically
    loaded.food_capacity_field.update(world.state.time + 300)
    world.food_capacity_field.update(world.state.time + 300)
    torch.testing.assert_close(loaded.food_capacity, world.food_capacity, rtol=0, atol=1e-6)
    loaded.step()


def test_static_field_never_changes_and_drifting_field_does(make_world):
    static = make_world(food_capacity_variation=0.5, food_capacity_scale=25.0)
    drifting = make_world(food_capacity_variation=0.5, food_capacity_scale=25.0,
                          food_capacity_change_steps=200, food_capacity_update_every=10)
    s0, d0 = static.food_capacity.clone(), drifting.food_capacity.clone()
    assert not static.food_capacity_field.moving and drifting.food_capacity_field.moving
    for _ in range(25):
        static.step(); drifting.step()
    assert torch.equal(static.food_capacity, s0)
    changed = (drifting.food_capacity - d0).abs().max().item()
    assert changed > 0.05
    # it is the same tensor the grow kernel reads, updated in place
    assert drifting.food_capacity.data_ptr() == drifting.food_capacity_field.multiplier.data_ptr()


def test_drifting_field_decorrelates_over_change_steps(make_world):
    world = make_world(food_capacity_variation=0.5, food_capacity_scale=25.0,
                       food_capacity_change_steps=1000)
    field, p = world.food_capacity_field, world.creatures.pad
    def at(t):
        field.update(t)
        return field.multiplier[p:-p, p:-p].flatten().clone()
    base = at(0)
    corr = lambda t: torch.corrcoef(torch.stack([base, at(t)]))[0, 1].item()
    assert corr(20) > 0.95       # slow: barely changes over a few dozen steps
    assert corr(1000) < 0.5      # but is largely different after change_steps
