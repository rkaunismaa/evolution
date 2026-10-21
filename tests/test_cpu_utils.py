"""Pure-Python/CPU pieces: these run without a GPU."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evolution.core.config import ConfigFunction, Config
from evolution.state.metric_tracker import MetricTracker
from evolution.utils.quantize import quantize
from evolution.utils.subscribe import Publisher, Subscriber


class Counter(Subscriber):
    def __init__(self, freq):
        super().__init__(freq)
        self.calls = []

    def _update(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_subscriber_updates_only_after_freq_notifications():
    pub, sub = Publisher(), Counter(freq=3)
    pub.subscribe(sub)
    for i in range(7):
        pub.publish(i, tag='x')
        pub.update_all()
    # fires after notifications 3 and 6, with the latest arguments each time
    assert sub.calls == [((2,), {'tag': 'x'}), ((5,), {'tag': 'x'})]


def test_publisher_rejects_duplicates_and_init_publish_forces_update():
    pub, slow, fast = Publisher(), Counter(freq=10), Counter(freq=2)
    pub.subscribe(slow)
    pub.subscribe(fast)
    with pytest.raises(ValueError):
        pub.subscribe(slow)
    pub.init_publish()
    pub.update_all()
    assert len(slow.calls) == 1 and len(fast.calls) == 1
    pub.unsubscribe(slow)
    assert pub.subscribers == [fast]


def _fake_state():
    return SimpleNamespace(game_step_publisher=Publisher(), time=0)


def test_metric_tracker_scalar_values_grow_past_initial_buffer():
    state = _fake_state()
    values = iter(range(2500))
    tracker = MetricTracker(state, lambda: float(next(values)), poll_rate=1)
    for t in range(2500):   # more than the 1000-sample growth chunk
        state.time = t
        state.game_step_publisher.publish()
        state.game_step_publisher.update_all()
    assert tracker.length == 2500
    np.testing.assert_array_equal(tracker.times, np.arange(2500))
    np.testing.assert_array_equal(tracker.values, np.arange(2500))
    assert (tracker.min_val, tracker.max_val) == (0, 2499)
    assert tracker.has_updates() is True and tracker.has_updates() is False


def test_metric_tracker_array_values_and_state_dict_round_trip():
    state = _fake_state()
    tracker = MetricTracker(state, lambda n: np.arange(n) * state.time, poll_rate=2, n=4)
    for t in range(1, 11):
        state.time = t
        state.game_step_publisher.publish()
        state.game_step_publisher.update_all()
    assert tracker.values.shape == (4, 5)          # polled every 2 steps
    np.testing.assert_array_equal(tracker.values[:, -1], np.arange(4) * 10)

    other = MetricTracker(_fake_state(), lambda n: None, poll_rate=2, n=4)
    other.load_state_dict(tracker.state_dict())
    np.testing.assert_array_equal(other.values, tracker.values)
    np.testing.assert_array_equal(other.times, tracker.times)
    assert other.length == tracker.length


@pytest.mark.parametrize("shape", [(500,), (500, 3), (200, 8, 5)])
def test_quantize_round_trip_error_is_within_one_step(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape) * 3 + 1
    q = quantize(x)
    y = q.dequantize(map_location=torch.device('cpu'))
    assert y.shape == x.shape
    flat = x.view(x.shape[0], -1)
    span = flat.amax(0).clamp(min=0) - flat.amin(0).clamp(max=0)
    err = (y.view_as(flat) - flat).abs()
    assert (err <= span / 255 + 1e-6).all()


def test_quantize_accepts_float16():
    x = torch.randn(100, 6).half()
    y = quantize(x).dequantize(map_location=torch.device('cpu'))
    assert y.dtype == torch.float32
    assert (y - x.float()).abs().max() < 0.1


def test_config_functions():
    sq = ConfigFunction('square_add', 11., 1.0)
    assert sq(torch.tensor(2.0)).item() == pytest.approx(45.0)
    assert ConfigFunction('linear_frac', 0.3)(torch.tensor(2.0), torch.tensor(1.0)).item() == pytest.approx(0.3)
    assert ConfigFunction('abs_exp', 0.2)(torch.tensor(0.0), torch.tensor(5.0)).item() == 0.0
    assert repr(sq) == 'square_add((11.0, 1.0))'


def test_config_update_in_place():
    cfg = Config()
    cfg.update_in_place({'size': 77, 'kill_reproduce_every': 4})
    assert (cfg.size, cfg.kill_reproduce_every) == (77, 4)
    other = Config(brain_dtype='float16')
    cfg.update_in_place(other)
    assert cfg.brain_dtype == 'float16' and cfg.size == other.size
