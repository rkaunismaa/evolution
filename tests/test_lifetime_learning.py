"""Lifetime learning (Config.lifetime_learning = 1): reward-modulated Hebbian last-layer weights."""
import pickle

import pytest
import torch

from tests.conftest import requires_cuda

pytestmark = requires_cuda

LEARNING_TRAITS = ('learning_rates', 'learned_weights', 'last_hidden', 'last_outputs',
                   'last_energies', 'reward_avgs', 'reward_spreads')


def test_off_by_default(make_world):
    traits = make_world().creatures.traits.variables
    assert not any(name in traits for name in LEARNING_TRAITS)


def test_traits_and_shapes(make_world):
    world = make_world(lifetime_learning=1)
    c = world.creatures
    assert c.learned_weights._shape == (c.weights[-1]._shape[0], 2)   # default: move and rotate learn
    world = make_world(lifetime_learning=1, learning_outputs=0, learning_noise=0.0, learning_trace_decay=0.0)
    c = world.creatures
    hidden, n_out = c.weights[-1]._shape
    assert c.learned_weights._shape == (hidden, n_out)
    assert c.learned_weights.data.eq(0).all()
    lo, hi = world.cfg.init_learning_rate_range
    assert (c.learning_rates.data >= lo).all() and (c.learning_rates.data <= hi).all()
    assert torch.equal(c.last_energies.data, c.energies.data)   # first reward is 0


def _one_update(c, reward):
    n = c.population
    c.learned_weights.data[:] = 0.0
    c.reward_avgs.data[:] = 0.0
    c.reward_spreads.data[:] = 1.0
    c.learning_rates.data[:] = 0.1
    c.last_hidden.data[:] = torch.linspace(-1, 1, c.last_hidden._shape[0], device='cuda')
    c.last_outputs.data[:] = torch.linspace(1, -1, c.last_outputs._shape[0], device='cuda')
    c.last_energies.data[:] = c.energies.data - reward
    c.learn()
    return torch.outer(c.last_hidden.data[0], c.last_outputs.data[0])


@pytest.mark.parametrize("reward", [0.5, -0.5])
def test_hebbian_update_follows_the_reward(make_world, reward):
    world = make_world(lifetime_learning=1, learning_outputs=0, learning_noise=0.0, learning_trace_decay=0.0)
    c = world.creatures
    hebb = _one_update(c, reward)
    expected = 0.1 * reward * hebb      # advantage = reward / spread with avg 0, spread 1
    torch.testing.assert_close(c.learned_weights.data[0], expected, rtol=1e-4, atol=1e-6)
    assert torch.equal(c.last_energies.data, c.energies.data)


def test_learned_weights_are_clamped(make_world):
    world = make_world(lifetime_learning=1, learning_outputs=0, learning_noise=0.0, learning_trace_decay=0.0, learned_weight_clip=0.01)
    c = world.creatures
    _one_update(c, 4.0)
    assert c.learned_weights.data.abs().max().item() == pytest.approx(0.01)


def test_forward_uses_learned_weights(make_world):
    world = make_world(lifetime_learning=1, learning_outputs=0, learning_noise=0.0, learning_trace_decay=0.0)
    c = world.creatures
    world.step()
    stimuli = c.collect_stimuli(world.collisions, world.food_grid).unsqueeze(1)
    mem = c.memories.data.clone()
    base = c.forward(stimuli)
    c.memories.data[:] = mem
    c.learned_weights.data[:] = 0.3
    learned = c.forward(stimuli)
    hidden = c.activations[-2]
    w, b = c.weights[-1].data, c.biases[-1].data
    expected = torch.tanh(hidden @ w.float() + b + hidden @ c.learned_weights.data).squeeze(1)
    torch.testing.assert_close(learned, expected, rtol=1e-3, atol=1e-3)
    assert not torch.allclose(base, learned)
    torch.testing.assert_close(c.last_outputs.data, learned)


def test_children_start_unlearned_and_inherit_the_learning_rate(make_world):
    world = make_world(lifetime_learning=1)
    c = world.creatures
    c.learned_weights.data[:] = 0.5
    parents = torch.arange(10, device='cuda')
    mut = torch.zeros_like(c.mutation_rates.data[parents])
    kids = c.traits.reproduce_traits(mut, parents, 10)
    assert kids.variables['learned_weights'].data.eq(0).all()
    assert torch.equal(kids.variables['learning_rates'].data, c.learning_rates.data[parents])


def test_paying_for_a_child_is_not_a_negative_reward(make_world):
    world = make_world(lifetime_learning=1, start_creatures=64)
    c = world.creatures
    world.step()
    c.ages.data[:] = 1e4
    c.energies.data[:] = 1e3
    c.healths.data[:] = 1e3
    c.last_energies.data[:] = c.energies.data
    pop = c.population
    c.fused_kill_reproduce(world.central_food_grid, world.state)
    assert c.n_births == pop
    parents = c.n_children.data == 1
    gap = (c.energies.data - c.last_energies.data)[parents]
    assert gap.abs().max().item() < 1e-3       # the energy spent on children cancels out


def test_invariants_and_checkpoint_with_learning(make_world, tmp_path):
    from evolution.core.gworld import GWorld
    world = make_world(lifetime_learning=1, kill_reproduce_every=4, brain_dtype='float16')
    c = world.creatures
    for step in range(200):
        assert world.step(), f"mass extinction at step {step}"
    assert c.learned_weights.data.abs().sum() > 0
    for name, v in c.traits.variables.items():
        assert torch.isfinite(v.data.float()).all(), name
    path = tmp_path / "learn.ckpt"
    world.write_checkpoint(str(path), quantized=False)
    loaded, _ = GWorld.from_checkpoint(str(path))
    for name in LEARNING_TRAITS:
        assert torch.equal(loaded.creatures.traits.variables[name].data, c.traits.variables[name].data), name
    loaded.step()


def test_older_configs_do_not_learn():
    from evolution.core.config import Config
    cfg = Config(lifetime_learning=1)
    del cfg.__dict__['lifetime_learning']
    assert pickle.loads(pickle.dumps(cfg)).lifetime_learning == 0


def test_only_the_first_outputs_learn(make_world):
    world = make_world(lifetime_learning=1, learning_outputs=2, learning_noise=0.0)
    c = world.creatures
    hidden = c.weights[-1]._shape[0]
    assert c.learned_weights._shape == (hidden, 2) and c.last_outputs._shape == (2,)
    world.step()
    stimuli = c.collect_stimuli(world.collisions, world.food_grid).unsqueeze(1)
    mem = c.memories.data.clone()
    base = c.forward(stimuli)
    c.memories.data[:] = mem
    c.learned_weights.data[:] = 0.3
    learned = c.forward(stimuli)
    assert not torch.allclose(base[:, :2], learned[:, :2])     # move and rotate change
    torch.testing.assert_close(base[:, 2:], learned[:, 2:])    # memory outputs do not


def test_trace_credits_earlier_steps(make_world):
    world = make_world(lifetime_learning=1, learning_trace_decay=0.5, learning_noise=0.0)
    c = world.creatures
    n_hidden = c.last_hidden._shape[0]
    c.reward_avgs.data[:] = 0.0
    c.reward_spreads.data[:] = 1.0
    c.learning_rates.data[:] = 0.1
    c.learned_weights.data[:] = 0.0
    c.learning_traces.data[:] = 0.0
    # step 1: activity, no reward
    c.last_hidden.data[:] = 1.0
    c.last_outputs.data[:] = 1.0
    c.last_energies.data[:] = c.energies.data
    c.learn()
    assert c.learned_weights.data.abs().max() == 0
    # step 2: no activity, but a reward: the trace still credits step 1's activity
    c.last_hidden.data[:] = 0.0
    c.last_outputs.data[:] = 0.0
    c.reward_avgs.data[:] = 0.0
    c.reward_spreads.data[:] = 1.0
    c.last_energies.data[:] = c.energies.data - 1.0
    c.learn()
    # trace after step 1: 0.5 * 1 = 0.5; after step 2: 0.5 * 0.5 = 0.25; update 0.1 * 1 * 0.25
    torch.testing.assert_close(c.learned_weights.data[0], torch.full((n_hidden, c.n_learn), 0.025, device='cuda'))


def test_exploration_noise_drives_the_update(make_world):
    """learning_noise > 0: the learning outputs are perturbed, and the update follows the
    perturbation (node perturbation) rather than the outputs."""
    world = make_world(lifetime_learning=1, learning_outputs=2, learning_noise=0.5, learning_trace_decay=0.0)
    c = world.creatures
    world.step()
    stimuli = c.collect_stimuli(world.collisions, world.food_grid).unsqueeze(1)
    mem = c.memories.data.clone()
    out1 = c.forward(stimuli)
    noise1 = c.last_outputs.data.clone()
    c.memories.data[:] = mem
    out2 = c.forward(stimuli)
    assert not torch.allclose(out1[:, :2], out2[:, :2])          # move and rotate are noisy
    torch.testing.assert_close(out1[:, 2:], out2[:, 2:])         # memory outputs are not
    assert 0.4 < noise1.std().item() < 0.6                         # the stored noise, not the outputs
    # the update is lr * advantage * hidden x noise
    c.reward_avgs.data[:] = 0.0
    c.reward_spreads.data[:] = 1.0
    c.learning_rates.data[:] = 0.1
    c.learned_weights.data[:] = 0.0
    c.last_energies.data[:] = c.energies.data - 1.0
    hidden, noise = c.last_hidden.data.clone(), c.last_outputs.data.clone()
    c.learn()
    torch.testing.assert_close(c.learned_weights.data, 0.1 * hidden[:, :, None] * noise[:, None, :])
