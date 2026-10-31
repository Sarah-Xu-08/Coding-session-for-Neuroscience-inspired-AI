"""Real MPE2 Simple Spread with identical coverage rewards for every agent.

The two controllers below are transparent teaching baselines, not learned policies.
Both read global positions; the assignment controller is a centralized reference.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import linear_sum_assignment

COLORS = ['#238be6', '#ee6d9f', '#f5a623']
ACTION_NAMES = ['stay', 'left', 'right', 'down', 'up']


def make_spread(n_agents=3, max_cycles=60, render=False):
    from mpe2 import simple_spread_v3
    return simple_spread_v3.parallel_env(
        N=n_agents, local_ratio=0.0, max_cycles=max_cycles,
        continuous_actions=False, render_mode='rgb_array' if render else None)


def spread_positions(env):
    world = env.unwrapped.world
    return (np.array([a.state.p_pos for a in world.agents]),
            np.array([landmark.state.p_pos for landmark in world.landmarks]))


def reset_spread(env, seed=2026, layout='clustered'):
    observations, infos = env.reset(seed=seed)
    if layout == 'clustered':
        world = env.unwrapped.world
        if len(world.agents) != 3:
            raise ValueError('The clustered teaching layout has three agents; use layout="random" otherwise.')
        agents = np.array([[-.95, -.2], [-.45, -.4], [-.75, -.9]])
        landmarks = np.array([[-.75, -.55], [.75, -.55], [0., .75]])
        for agent, position in zip(world.agents, agents):
            agent.state.p_pos = position.copy()
            agent.state.p_vel = np.zeros(2)
        for landmark, position in zip(world.landmarks, landmarks):
            landmark.state.p_pos = position.copy()
        observations = {name: env.unwrapped.observe(name) for name in env.agents}
    elif layout != 'random':
        raise ValueError('Choose clustered or random.')
    return observations, infos


def landmark_targets(agent_positions, landmark_positions, strategy):
    distances = np.linalg.norm(agent_positions[:, None, :] - landmark_positions[None, :, :], axis=-1)
    if strategy == 'nearest':
        return distances.argmin(axis=1)
    if strategy == 'assignment':
        agents, landmarks = linear_sum_assignment(distances)
        targets = np.empty(len(agents), dtype=int)
        targets[agents] = landmarks
        return targets
    raise ValueError('Choose nearest or assignment.')


def movement_action(position, velocity, target):
    # Discrete, damped steering; physical collisions and inertia remain enabled.
    error = target - position - .35 * velocity
    axis = int(np.argmax(np.abs(error)))
    if abs(error[axis]) < .045:
        return 0
    return (2 if error[0] > 0 else 1) if axis == 0 else (4 if error[1] > 0 else 3)


def spread_rollout(strategy='assignment', seed=2026, layout='clustered', steps=60):
    env = make_spread(max_cycles=steps)
    try:
        observations, _ = reset_spread(env, seed, layout)
        names = list(env.agents)
        agents, landmarks = spread_positions(env)
        targets = landmark_targets(agents, landmarks, strategy)
        positions = [agents.copy()]
        initial_reward = float(env.unwrapped.scenario.global_reward(env.unwrapped.world))
        scores = [initial_reward]
        records = []
        for step in range(steps):
            agents, _ = spread_positions(env)
            if strategy == 'nearest':
                targets = landmark_targets(agents, landmarks, strategy)
            velocities = [a.state.p_vel for a in env.unwrapped.world.agents]
            actions = {name: movement_action(agents[i], velocities[i], landmarks[targets[i]])
                       for i, name in enumerate(names)}
            observations, rewards, terminated, truncated, _ = env.step(actions)
            values = np.array([rewards[name] for name in names])
            if not np.allclose(values, values[0]):
                raise AssertionError('Simple Spread must return the same team reward to every agent.')
            agents, _ = spread_positions(env)
            positions.append(agents.copy())
            scores.append(float(values[0]))
            records.append(dict(step=step + 1, team_reward=float(values[0]),
                                **{f'reward_{name}': rewards[name] for name in names}))
            if all(terminated[name] or truncated[name] for name in names):
                break
        return dict(strategy=strategy, seed=seed, layout=layout, landmarks=landmarks,
                    positions=np.array(positions), rewards=np.array(scores),
                    history=pd.DataFrame(records), observation_shape=observations[names[0]].shape)
    finally:
        env.close()


def spread_bounds(*rollouts):
    extent = np.concatenate([points for rollout in rollouts for points in
                             (rollout['positions'].reshape(-1, 2), rollout['landmarks'])])
    return extent.min(axis=0) - .28, extent.max(axis=0) + .28


def _scene(ax, rollout, frame, trails=True, bounds=None):
    points = rollout['positions'][frame]
    landmarks = rollout['landmarks']
    low, high = spread_bounds(rollout) if bounds is None else bounds
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), aspect='equal',
           xlabel='x position (world units)', ylabel='y position (world units)')
    ax.set_facecolor('#edf8f2')
    ax.grid(alpha=.15)
    for j, landmark in enumerate(landmarks):
        ax.add_patch(Circle(landmark, .1, fc='#43bc95', alpha=.2, ec='none'))
        ax.scatter(*landmark, marker='X', s=120, color='#196447', zorder=3)
        ax.annotate(f'L{j + 1}', landmark, xytext=(8, 8), textcoords='offset points', color='#196447')
    for i, point in enumerate(points):
        if trails:
            trail = rollout['positions'][:frame + 1, i]
            ax.plot(trail[:, 0], trail[:, 1], color=COLORS[i], alpha=.6, lw=2)
        ax.add_patch(Circle(point, .075, fc=COLORS[i], ec='white', lw=1.5, zorder=5))
        ax.text(*point, str(i), ha='center', va='center', color='white', weight='bold', zorder=6)


def plot_spread_comparison(nearest, assigned, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    bounds = spread_bounds(nearest, assigned)
    for ax, rollout, title in zip(axes[:2], [nearest, assigned],
                                  ['Everyone chooses nearest', 'One landmark per agent']):
        _scene(ax, rollout, len(rollout['positions']) - 1, bounds=bounds)
        ax.set_title(title)
    for rollout, label, color in [(nearest, 'Nearest landmark', '#e16c63'),
                                  (assigned, 'Centralized assignment', '#238be6')]:
        axes[2].plot(rollout['history'].step, rollout['history'].team_reward, label=label, color=color, lw=2)
    axes[2].set(xlabel='Environment steps', ylabel='Shared reward per step',
                title='Same score for all three agents')
    axes[2].legend(frameon=False, fontsize=9)
    if save_path is not None:
        fig.savefig(save_path, dpi=180)
    plt.show()


def plot_spread_frame(rollout, frame, reward_fn, bounds=None):
    fig, ax = plt.subplots(figsize=(5.5, 4.8), layout='constrained')
    _scene(ax, rollout, frame, bounds=bounds)
    reward = float(reward_fn(rollout['positions'][frame], rollout['landmarks']))
    if not np.isclose(reward, rollout['rewards'][frame]):
        raise AssertionError('Your reward should match the actual environment reward.')
    ax.set_title(f"{rollout['strategy']} · step {frame}\n"
                 f'r₀ = r₁ = r₂ = {reward:.3f}')
    plt.show()
    plt.close(fig)
