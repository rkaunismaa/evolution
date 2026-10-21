"""Memory management in CreatureArray.add_with_deaths: after any mix of deaths and births, the
current block must hold exactly the survivors plus the children, with every trait of a creature
kept together in the same row, and the selected creature must still point at the same creature."""
import pytest
import torch

from tests.conftest import requires_cuda

pytestmark = requires_cuda

# traits we tag with a creature id, covering scalar, vector and matrix traits
def _tagged(creatures):
    return {
        'ages': lambda d: d,
        'positions': lambda d: d[:, 0],
        'memories': lambda d: d[:, 0],
        'weights_0': lambda d: d[:, 0, 0],
    }


def _write_ids(arr_vars, ids):
    """Write ids into each tagged trait of a CreatureArray's variables (current memory)."""
    arr_vars['ages'].data[:] = ids
    arr_vars['positions'].data[:, 0] = ids
    arr_vars['memories'].data[:, 0] = ids
    arr_vars['weights_0'].data[:, 0, 0] = ids


def _read_ids(arr_vars):
    ages = arr_vars['ages'].data
    others = [arr_vars['positions'].data[:, 0], arr_vars['memories'].data[:, 0],
              arr_vars['weights_0'].data[:, 0, 0]]
    for o in others:   # every trait in a row must belong to the same creature
        assert torch.equal(o, ages), "traits of one creature ended up in different rows"
    return ages


def _check_block(traits):
    """alive mask covers exactly [start_idx, start_idx + population)."""
    pop, start = traits.population, traits.start_idx
    assert traits.alive[start:start + pop].eq(1).all()
    assert traits.alive.sum().item() == pop
    for v in traits.variables.values():
        assert v.data.shape[0] == pop


def _one_update(world, n_dead, n_born, next_id, select=None):
    """Kill n_dead random creatures and add n_born children of random survivors.
    Returns (expected id set, next_id, selected id or None)."""
    c, traits = world.creatures, world.creatures.traits
    pop = c.population
    perm = torch.randperm(pop, device=c.device)
    dead = torch.zeros(pop, dtype=torch.bool, device=c.device)
    dead[perm[:n_dead]] = True
    alive = ~dead
    dead_idxs = dead.nonzero().squeeze(1)

    ids = _read_ids(traits.variables)
    survivors = ids[alive]

    children = None
    child_ids = torch.empty(0, device=c.device)
    if n_born > 0:
        parents = alive.nonzero().squeeze(1)[torch.randint(int(alive.sum()), (n_born,), device=c.device)]
        children = traits.reproduce_traits(c.mutation_rates[parents], parents, n_born)
        child_ids = torch.arange(next_id, next_id + n_born, device=c.device, dtype=torch.float32)
        _write_ids(children.variables, child_ids)
        next_id += n_born

    selected_id = None
    if select is not None:
        world.state.selected_creature = select
        selected_id = ids[select].item()
        if not alive[select]:
            selected_id = None

    traits.add_with_deaths(dead_idxs, alive, n_dead, children, world.state)

    expected = torch.cat([survivors, child_ids])
    return expected, next_id, selected_id


def _setup(make_world, **cfg):
    world = make_world(**cfg)
    c = world.creatures
    _write_ids(c.traits.variables, torch.arange(c.population, device=c.device, dtype=torch.float32))
    return world, c.population


@pytest.mark.parametrize("case", ["fill_gaps", "move_block", "max"])
def test_each_strategy_keeps_creatures_intact(make_world, case):
    world, pop = _setup(make_world)
    max_c = world.cfg.max_creatures
    if case == "fill_gaps":       # more births than deaths
        n_dead, n_born = 50, 120
    elif case == "move_block":    # more deaths than births
        n_dead, n_born = 200, 30
    else:                         # fill the buffer exactly
        n_dead, n_born = 10, max_c - pop + 10

    expected, _, _ = _one_update(world, n_dead, n_born, next_id=pop)
    traits = world.creatures.traits
    got = _read_ids(traits.variables)
    assert torch.equal(got.sort().values, expected.sort().values)
    assert traits.algos[case] == 1
    _check_block(traits)


def test_filling_buffer_when_block_does_not_start_at_zero(make_world):
    """Reaching exactly max_creatures after the block has moved away from index 0 must use the
    free slots on the left of the block too, and keep the selected creature pointing at itself."""
    world, pop = _setup(make_world, start_creatures=1024, max_creatures=2048)
    traits = world.creatures.traits
    next_id = pop
    # deaths outnumber births -> move_block, which normally leaves the block away from index 0
    for _ in range(5):
        _, next_id, _ = _one_update(world, 300, 10, next_id)
        if traits.start_idx > 0:
            break
    assert traits.start_idx > 0, "setup failed to move the block"

    pop = traits.population
    select = pop // 2
    n_dead = 20
    n_born = world.cfg.max_creatures - pop + n_dead   # exactly fills the buffer
    expected, _, selected_id = _one_update(world, n_dead, n_born, next_id, select=select)
    got = _read_ids(traits.variables)
    assert traits.population == world.cfg.max_creatures
    assert torch.equal(got.sort().values, expected.sort().values)
    _check_block(traits)
    if selected_id is not None:
        assert got[int(world.state.selected_creature)].item() == selected_id


def test_random_sequence_of_births_and_deaths(make_world):
    """Fuzz: many updates in a row, hitting every strategy, with a nonzero block start."""
    world, pop = _setup(make_world, start_creatures=1024, max_creatures=2048)
    traits = world.creatures.traits
    gen = torch.Generator().manual_seed(1)
    next_id = pop
    for _ in range(60):
        pop = traits.population
        room = world.cfg.max_creatures - pop
        n_dead = int(torch.randint(0, max(1, pop // 3), (1,), generator=gen))
        if torch.rand(1, generator=gen).item() < 0.2:   # sometimes fill the buffer exactly
            n_born = room + n_dead
        else:
            n_born = int(torch.randint(0, min(room + n_dead, pop) + 1, (1,), generator=gen))
        if n_dead == 0 and n_born == 0:
            continue
        expected, next_id, _ = _one_update(world, n_dead, n_born, next_id)
        got = _read_ids(traits.variables)
        assert torch.equal(got.sort().values, expected.sort().values)
        _check_block(traits)
    assert all(v > 0 for v in traits.algos.values()), traits.algos


@pytest.mark.parametrize("n_dead,n_born", [(50, 120), (200, 30), (300, 0)])
def test_selected_creature_is_tracked(make_world, n_dead, n_born):
    for where in ("first", "middle", "last"):
        world, pop = _setup(make_world)
        select = {"first": 0, "middle": pop // 2, "last": pop - 1}[where]
        _, _, selected_id = _one_update(world, n_dead, n_born, next_id=10**6, select=select)
        sel = world.state.selected_creature
        if selected_id is None:
            assert not sel   # it died, so the selection is cleared
        else:
            ids = _read_ids(world.creatures.traits.variables)
            assert ids[int(sel)].item() == selected_id, where
