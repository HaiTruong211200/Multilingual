import math

import torch

from src.layer_scheduler import POTSAStyleLayerScheduler


def test_reward_and_ema_update():
    scheduler = POTSAStyleLayerScheduler([8], ema_rho=0.1)
    assert scheduler.observe(8, 1.2) is None
    assert math.isclose(scheduler.observe(8, 1.0), 0.2)
    assert math.isclose(scheduler.states[8].q_value, 0.02)
    assert scheduler.states[8].count == 2


def test_ucb_prefers_less_visited_layer():
    scheduler = POTSAStyleLayerScheduler([8, 10])
    scheduler.states[8].count = 100
    scheduler.states[10].count = 2
    utilities = scheduler.get_utilities(100)
    assert utilities[10] > utilities[8]


def test_probabilities_are_valid():
    scheduler = POTSAStyleLayerScheduler([8, 10, 12])
    probabilities = scheduler.get_probabilities(10)
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert all(value >= 0.0 for value in probabilities.values())


def test_force_each_layer_once():
    scheduler = POTSAStyleLayerScheduler([8, 10, 12], force_each_layer_once=True)
    selected = []
    for step in range(3):
        layer = scheduler.current_layer
        selected.append(layer)
        scheduler.observe(layer, 1.0)
        scheduler.sample_next(step + 1)
    assert selected == [8, 10, 12]


def test_state_round_trip_preserves_sampling():
    scheduler = POTSAStyleLayerScheduler([8, 10], seed=7)
    scheduler.observe(8, 1.2)
    scheduler.observe(8, 1.0)
    state = scheduler.state_dict()

    restored = POTSAStyleLayerScheduler([8, 10], seed=999)
    restored.load_state_dict(state)
    assert restored.states == scheduler.states
    assert restored.current_layer == scheduler.current_layer
    assert torch.equal(restored.generator.get_state(), scheduler.generator.get_state())
    assert restored.sample_next(100) == scheduler.sample_next(100)
