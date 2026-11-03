"""Sexual reproduction (Config.sexual_reproduction = 1): mate finding, crossover and the full loop."""
import pickle

import pytest
import torch

from tests.conftest import requires_cuda

pytestmark = requires_cuda


def _place(c, xy, colors):
    """Put the first len(xy) creatures at the given positions and colours."""
    n = len(xy)
    c.positions.data[:n] = torch.tensor(xy, device='cuda', dtype=torch.float32)
    c.colors.data[:n] = torch.tensor(colors, device='cuda', dtype=torch.float32)
    return torch.arange(n, device='cuda')


def test_mates_must_share_a_square_and_be_close_in_colour(make_world):
    world = make_world(sexual_reproduction=1, mate_range=20.0, mate_color_range=20.0)
    c = world.creatures
    idx = _place(c, [(5, 5), (8, 9),        # same square, similar colours: a pair
                     (45, 5), (50, 7),      # same square, colours too far apart
                     (19, 30), (21, 30),    # close, but on either side of a square boundary
                     (70, 70)],             # alone
                 [100, 110, 30, 200, 60, 61, 90])
    a, b = c._find_mates(idx)
    pairs = {tuple(sorted(p)) for p in zip(a.tolist(), b.tolist())}
    assert pairs == {(0, 1)}


def test_each_creature_mates_at_most_once(make_world):
    world = make_world(sexual_reproduction=1, mate_range=50.0, mate_color_range=255.0)
    c = world.creatures
    idx = _place(c, [(10 + i * 0.1, 10) for i in range(7)], [50 + i for i in range(7)])
    a, b = c._find_mates(idx)
    together = torch.cat([a, b])
    assert a.shape[0] == 3                                  # 7 in one square: 3 pairs, 1 left over
    assert together.unique().shape[0] == together.shape[0]
    assert torch.equal((c.colors.data[b] - c.colors.data[a]).abs(), torch.ones(3, device='cuda'))  # neighbours in colour


def test_crossover_masks_keep_neurons_whole(make_world):
    world = make_world(sexual_reproduction=1)
    traits = world.creatures.traits
    masks = traits.crossover_masks(16)
    n_layers = len(world.creatures.weights)
    for i in range(n_layers):
        w, b = traits.variables[f'weights_{i}'], traits.variables[f'biases_{i}']
        assert masks[f'weights_{i}'] is masks[f'biases_{i}']         # one mask per layer
        assert masks[f'weights_{i}'].shape == (16, 1, w._shape[-1])   # one entry per output neuron
        assert b._shape[-1] == w._shape[-1]
    assert masks['sizes'].shape == (16,)
    assert masks['rays'].shape == (16, world.cfg.num_rays, 1)          # whole rays
    assert 'positions' not in masks                                  # children sit near the paying parent
    assert 'mutation_rates' not in masks                             # inherited whole from the paying parent


def test_children_take_each_neuron_from_one_parent(make_world):
    world = make_world(sexual_reproduction=1)
    c = world.creatures
    mothers = torch.arange(0, 20, device='cuda')
    fathers = torch.arange(20, 40, device='cuda')
    mut = torch.zeros_like(c.mutation_rates.data[mothers])          # no mutation: pure crossover
    kids = c.traits.reproduce_traits(mut, mothers, 20, fathers)
    from_father_somewhere = False
    for i in range(len(c.weights)):
        kw = kids.variables[f'weights_{i}'].data.float()
        kb = kids.variables[f'biases_{i}'].data.float()
        mw, fw = c.weights[i].data[mothers].float(), c.weights[i].data[fathers].float()
        mb, fb = c.biases[i].data[mothers].float(), c.biases[i].data[fathers].float()
        col_from_m = (kw == mw).all(1)                               # [20, out]: whole column matches
        col_from_f = (kw == fw).all(1)
        assert (col_from_m | col_from_f).all()
        # the bias of each neuron comes from the same parent as its weights
        bias_m, bias_f = (kb == mb)[:, 0], (kb == fb)[:, 0]
        assert ((col_from_m & bias_m) | (col_from_f & bias_f)).all()
        from_father_somewhere |= bool((col_from_f & ~col_from_m).any())
    assert from_father_somewhere
    ks = kids.variables['sizes'].data
    assert ((ks == c.sizes.data[mothers]) | (ks == c.sizes.data[fathers])).all()
    assert torch.equal(kids.variables['mutation_rates'].data, c.mutation_rates.data[mothers])


def test_each_pair_has_two_children_paid_for_by_each_parent(make_world):
    world = make_world(sexual_reproduction=1, mate_range=1000.0, mate_color_range=255.0,
                       start_creatures=64)
    c = world.creatures
    world.step()
    c.ages.data[:] = 1e4                   # everyone mature,
    c.energies.data[:] = 1e3               # well fed,
    c.healths.data[:] = 1e3                # and healthy
    pop = c.population
    c.fused_kill_reproduce(world.central_food_grid, world.state)
    assert c.n_births == pop               # one big square, any colour: everyone pairs (64 is even)
    assert c.population == 2 * pop
    n_kids = c.n_children.data
    assert n_kids.sum().item() == pop and n_kids.max().item() == 1   # every parent paid for one child


def test_unpaired_creatures_wait(make_world):
    world = make_world(sexual_reproduction=1, mate_range=1.0, mate_color_range=0.0, start_creatures=64)
    c = world.creatures
    world.step()
    c.ages.data[:] = 1e4
    c.energies.data[:] = 1e3
    c.healths.data[:] = 1e3
    energies = c.energies.data.clone()
    c.fused_kill_reproduce(world.central_food_grid, world.state)
    assert c.n_births == 0 and torch.equal(c.energies.data, energies)


def test_invariants_with_sexual_reproduction(make_world):
    world = make_world(sexual_reproduction=1, mate_range=40.0, mate_color_range=60.0,
                       kill_reproduce_every=4, brain_dtype='float16')
    c = world.creatures
    for step in range(200):
        if step == 20:
            c.ages.data[:] = 1e4
            c.energies.data[:] = 100.
        assert world.step(), f"mass extinction at step {step}"
    assert c.n_births > 0
    for name, v in c.traits.variables.items():
        assert torch.isfinite(v.data.float()).all(), name


def test_older_configs_reproduce_asexually():
    from evolution.core.config import Config
    cfg = Config(sexual_reproduction=1)
    for name in ('sexual_reproduction', 'mate_range', 'mate_color_range'):
        del cfg.__dict__[name]
    old = pickle.loads(pickle.dumps(cfg))
    assert old.sexual_reproduction == 0
