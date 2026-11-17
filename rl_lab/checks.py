"""Small behavior checks for exercises; no reference implementations."""
import numpy as np
import torch


def check_foundations(epsilon_greedy, td_target, q_update):
    rng = np.random.default_rng(37)
    assert td_target(2., True, [100., 200.], [True, True], .95) == 2., 'A terminal state has no future value.'
    assert np.isclose(td_target(1., False, [4., 99.], [True, False], .5), 3.), 'Exclude invalid NEXT actions.'
    assert np.isclose(td_target(-2., False, [-4., -9.], [True, True], .5), -4.), 'The maximum can be negative.'
    assert np.isclose(q_update(1., 3., .25), 1.5), 'Check the learning-rate interpolation.'
    assert q_update(1., 3., 0.) == 1. and q_update(1., 3., 1.) == 3.
    for epsilon in [0., .5, 1.]:
        assert all(epsilon_greedy([100., -3.], epsilon, rng, [False, True]) == 1 for _ in range(20)), 'Both branches must respect the mask.'
    assert epsilon_greedy([1., 5., 2.], 0., rng) == 1, 'At epsilon=0 choose a maximum.'
    tied = {epsilon_greedy([2., 2., -1.], 0., rng) for _ in range(100)}
    assert tied == {0, 1}, 'Break ties randomly among maxima.'
    explored = {epsilon_greedy([9., 0., 1.], 1., rng) for _ in range(100)}
    assert explored == {0, 1, 2}, 'Exploration must reach all feasible actions.'
    print('Task 1 passed: TD targets and Q updates; provided exploration also checked.')


def check_dqn_targets(dqn_targets):
    rewards = torch.tensor([3., 1., -2.])
    terminated = torch.tensor([1., 0., 0.])
    y = dqn_targets(rewards, terminated, torch.tensor([100., .5, -4.]), .9)
    assert y.shape == (3,), 'Return one target per transition.'
    assert torch.allclose(y, torch.tensor([3., 1.45, -5.6])), 'Check termination masking and discounting.'
    print('Task 2 passed: one target per transition, terminal mask and discounting.')


def check_dqn_loss(dqn_loss):
    q = torch.tensor([[1., 3.], [4., 2.]], requires_grad=True)
    target = torch.tensor([2., 2.], requires_grad=True)
    loss = dqn_loss(q, torch.tensor([1, 0]), target)
    assert loss.ndim == 0 and torch.allclose(loss.detach(), torch.tensor(1.)), 'Use the batch mean Huber loss on the chosen actions.'
    loss.backward()
    assert torch.all(q.grad[torch.tensor([0, 1]), torch.tensor([0, 1])] == 0), 'Unchosen outputs should receive no direct loss gradient.'
    assert target.grad is None, 'The target must be detached.'
    print('Task 3 passed: chosen actions, scalar mean Huber loss and detached targets.')


def check_dqn(dqn_targets, dqn_loss):
    check_dqn_targets(dqn_targets)
    check_dqn_loss(dqn_loss)


def check_tit_for_tat(fn):
    assert [fn(x) for x in [2, 0, 1]] == [0, 0, 1], 'Start with cooperation; then copy the partner.'
    print('Task 4 passed: opening cooperation and replies to both actions.')


def check_team_reward(fn):
    landmarks = np.array([[0., 0.], [10., 0.]])
    cases = [(np.array([[1., 0.], [2., 0.]]), -9.),
             (np.array([[0., 0.], [0., 0.]]), -10.),
             (landmarks.copy(), 0.)]
    for agents, expected in cases:
        result = fn(agents, landmarks)
        assert np.asarray(result).shape == (), 'Return one scalar team score, not one score per agent.'
        assert np.isclose(result, expected), 'Measure nearest-agent distance for EACH LANDMARK, then sum and negate.'
    assert np.isclose(fn(np.array([[0., 0.]]), np.array([[3., 4.], [0., 0.]])), -5.), 'Use Euclidean distance in both coordinates.'
    agents = np.array([[0., 0.], [2., 1.]])
    landmarks = np.array([[0., 1.], [2., 2.], [4., 1.]])
    assert np.isclose(fn(agents, landmarks), -4.), 'Reduce over agents; do not reduce over landmarks.'
    assert np.isclose(fn(agents[::-1] + 7., landmarks[::-1] + 7.), -4.), 'Agent order, landmark order and a shared translation should not change coverage.'
    print('Task 5 passed: shared scalar, coverage direction, Euclidean distances and invariance.')
