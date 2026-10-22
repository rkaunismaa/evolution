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
    assert cap.min() >= 0.5 - 1e-5 and cap.max() <= 1.5 + 1e-5
    assert cap.max() - cap.min() > 0.5          # actually varies
    assert abs(cap.mean().item() - 1.0) < 0.1   # centred on the uniform capacity
    # smooth: neighbouring cells differ far less than the overall spread
    assert (cap[1:] - cap[:-1]).abs().max() < 0.1


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


@pytest.mark.parametrize("quantized", [False, True])
def test_food_capacity_map_saved_in_checkpoint(make_world, tmp_path, quantized):
    from evolution.core.gworld import GWorld
    world = make_world(food_capacity_variation=0.5, food_capacity_scale=25.0)
    world.step(); world.step()
    path = tmp_path / "cap.ckpt"
    world.write_checkpoint(str(path), quantized=quantized)
    loaded, _ = GWorld.from_checkpoint(str(path))
    tol = 0 if not quantized else 2.0 / 255
    assert (loaded.food_capacity - world.food_capacity).abs().max().item() <= tol
    loaded.step()
