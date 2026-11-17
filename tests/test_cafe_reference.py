"""Independent checks of the demand-only cookie reference and simulator."""
import numpy as np
import pytest
from rl_lab.cafe import CookieCafe, cafe_episode
from rl_lab.cafe_reference import solve_cafe_reference


@pytest.fixture(scope='module')
def reference():
    return solve_cafe_reference()


def test_reference_probabilities_and_reachability(reference):
    env = CookieCafe()
    assert reference.q.shape == (*env.observation_space.nvec, 3)
    assert reference.transitions.shape == (env.horizon, env.stock_cap + 1, env.stock_cap + 1)
    env.close()
    np.testing.assert_allclose(reference.transitions.sum(axis=-1), 1., atol=1e-12)
    assert reference.transitions.min() >= 0.
    assert reference.reachable[0].sum() == 1 and reference.reachable[0, 2]
    assert not reference.reachable[1, 7:].any()  # At most opening stock 2 plus preparation 4.
    assert reference.reachable[3].all()


def test_zero_demand_has_analytic_optimum():
    env = CookieCafe()
    env.MEANS = np.zeros_like(env.MEANS)
    reference = solve_cafe_reference(env)
    np.testing.assert_allclose(reference.transitions, np.broadcast_to(np.eye(env.stock_cap + 1), reference.transitions.shape))
    # No sales: do not prepare more; pay for and dispose of the two opening cookies.
    assert reference.expected_opening_return == pytest.approx(-6. - 2.)
    assert reference.act([0, 2], np.ones(3, dtype=bool)) == 0
    assert not reference.reachable[1, 0]


def test_last_hour_matches_analytic_poisson_sales():
    env = CookieCafe(missed_penalty=1.2, disposal_fee=2.)
    reference = solve_cafe_reference(env)
    rate = env.MEANS[-1]
    sold = 1. - np.exp(-rate)  # One available cookie; sell iff someone arrives.
    expected = sold * env.price - env.missed_penalty * (rate - sold)
    expected -= env.disposal_fee * (1. - sold)
    assert reference.values[env.horizon - 1, 1] == pytest.approx(expected)
    assert np.isneginf(reference.q[env.horizon - 1, 1, 1:]).all()


def test_exact_model_matches_actual_poisson_customers(reference):
    env = CookieCafe()
    t, stock, action = 2, 3, 1
    observed, rewards = np.zeros(9), []
    for seed in range(4000):
        env.reset(seed=seed)
        env.t, env.stock = t, stock
        state, reward, *_ = env.step(action)
        observed[state[1] - env.ACTIONS[action]] += 1
        rewards.append(reward)
    env.close()
    row = reference.transitions[t, stock]
    np.testing.assert_allclose(observed / len(rewards), row, atol=.025)
    sold = stock - row @ np.arange(9)
    expected = sold * env.price - env.missed_penalty * (env.MEANS[t] - sold)
    expected -= env.ACTIONS[action] * env.cost
    standard_error = np.std(rewards, ddof=1) / np.sqrt(len(rewards))
    assert abs(np.mean(rewards) - expected) < 6 * standard_error


def test_optimal_expected_return_matches_actual_nonclairvoyant_rollouts(reference):
    env = CookieCafe()
    returns = [cafe_episode(env, reference.act, seed)[0] for seed in range(30000, 30500)]
    standard_error = np.std(returns, ddof=1) / np.sqrt(len(returns))
    assert abs(np.mean(returns) - reference.expected_opening_return) < 6 * standard_error
    env.close()


def test_preparation_is_reliable_but_cannot_serve_current_customers():
    env = CookieCafe()
    for seed in range(10):
        state, _ = env.reset(seed=seed)
        assert tuple(state) == (0, 2)
        env.traffic[:] = 0
        env.traffic[0] = 5
        state, reward, *_rest, info = env.step(2)  # Prepare four cookies.
        assert info['sales'] == 2 and info['missed'] == 3
        assert info['produced'] == 4 and tuple(state) == (1, 4)
        assert reward == pytest.approx(12. - 12. - 6. - 1.5)
    env.close()


def test_demand_has_unequal_lunch_and_evening_peaks_with_evening_sales():
    env = CookieCafe()
    means = env.MEANS
    peaks = [10 + t for t in range(1, len(means) - 1)
             if means[t] > means[t - 1] and means[t] > means[t + 1]]
    assert peaks == [12, 18]
    assert means[12 - 10] > means[18 - 10] > 0
    _, rows = cafe_episode(env, lambda state, mask: 0, seed=2026)
    assert [r['hour'] for r in rows] == list(range(10, 20))
    assert rows[18 - 10]['demand'] == env.traffic[18 - 10]
    assert env.t == env.horizon == 10  # 20:00 closing, after the evening peak.
    env.close()
