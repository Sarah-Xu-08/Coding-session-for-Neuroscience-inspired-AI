"""One-product Dubai Chewy Cookie café, tabular learning, evaluation and brown result plots."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import gymnasium as gym
from gymnasium import spaces

COOKIE_BROWN = '#9a5b2f'


class CookieCafe(gym.Env):
    """State = (hour index, cookie stock). Only customer demand is random.

    Reserve capacity and prepare now; those cookies arrive after this hour's
    sales, ready for the next hour. Ingredients are always available. There is no
    cumulative ingredient budget, supplier failure, or second product.
    """
    metadata = {'render_modes': []}
    ACTIONS = np.array([0, 2, 4], dtype=int)
    # Hourly Poisson means for 10:00–19:00; closing is at 20:00.
    # A larger lunch peak at noon and a smaller evening peak at 18:00.
    MEANS = np.array([.5, 1.5, 4., 3., 1.5, .7, 1., 2., 3., 1.2])
    GAME_ID = 'dubai_cookie_cafe_v1'
    PRODUCT_NAME = 'Dubai Chewy Cookie'

    def __init__(self, stock_cap=8, opening_stock=2, price=6., cost=3.,
                 missed_penalty=.5, disposal_fee=1.):
        if not 0 <= opening_stock <= stock_cap:
            raise ValueError('Opening stock must be inside the inventory capacity.')
        self.stock_cap, self.opening_stock = stock_cap, opening_stock
        self.price, self.cost = price, cost
        self.missed_penalty, self.disposal_fee = missed_penalty, disposal_fee
        self.horizon = len(self.MEANS)
        self.observation_space = spaces.MultiDiscrete([self.horizon + 1, stock_cap + 1])
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.t, self.stock = 0, opening_stock

    def observation(self):
        return np.array([self.t, self.stock], dtype=np.int64)

    def action_mask(self):
        if self.t >= self.horizon - 1:
            return np.array([True, False, False])
        return self.stock + self.ACTIONS <= self.stock_cap

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.stock = 0, self.opening_stock
        # Hidden customer counts, generated before any decisions so equal seeds
        # produce the same customer day for every policy.
        self.traffic = self.np_random.poisson(self.MEANS)
        return self.observation(), {'action_mask': self.action_mask()}

    def step(self, action):
        if self.t >= self.horizon:
            raise RuntimeError('The café is closed; call reset().')
        if not self.action_space.contains(action) or not self.action_mask()[action]:
            raise ValueError(f'Action {action} is infeasible in state {self.observation()}.')
        old_stock = self.stock
        order = int(self.ACTIONS[action])
        demand = int(self.traffic[self.t])
        sold = min(old_stock, demand)
        missed = demand - sold
        # Preparation always succeeds, but cannot serve the current hour's customers.
        self.stock = old_stock - sold + order
        opening_cost = self.opening_stock * self.cost if self.t == 0 else 0.
        prep_cost = order * self.cost + opening_cost
        revenue = sold * self.price
        self.t += 1
        terminated = self.t == self.horizon
        waste = self.stock if terminated else 0
        disposal = waste * self.disposal_fee
        cash_profit = revenue - prep_cost - disposal
        reward = cash_profit - self.missed_penalty * missed
        info = dict(action_mask=self.action_mask(), inventory=old_stock,
                    produced=order, demand=demand, sales=sold, missed=missed,
                    waste=waste, cash_profit=cash_profit, revenue=revenue,
                    preparation_cost=prep_cost, disposal=disposal, next_inventory=self.stock)
        return self.observation(), float(reward), terminated, False, info


def cafe_model_parameters(env):
    return dict(stock_cap=env.stock_cap, opening_stock=env.opening_stock,
                price=env.price, cost=env.cost, missed_penalty=env.missed_penalty,
                disposal_fee=env.disposal_fee)


class CafeQAgent:
    def __init__(self, env, rules, seed=0, alpha=.15, gamma=1.):
        self.rules = rules
        self.q = np.zeros((*env.observation_space.nvec, env.action_space.n), np.float32)
        self.visits = np.zeros(tuple(env.observation_space.nvec), np.int32)
        self.rng = np.random.default_rng(seed)
        self.alpha, self.gamma = alpha, gamma
        self.env_kwargs = cafe_model_parameters(env)

    def act(self, state, mask, epsilon=0.):
        return self.rules.epsilon_greedy(self.q[tuple(state)], epsilon, self.rng, mask)

    def update(self, state, action, reward, next_state, terminated, next_mask):
        key = tuple(state)
        target = self.rules.td_target(reward, terminated, self.q[tuple(next_state)], next_mask, self.gamma)
        self.q[(*key, action)] = self.rules.q_update(self.q[(*key, action)], target, self.alpha)
        self.visits[key] += 1


def cafe_rule(state, mask, env):
    """A transparent forecast-based heuristic, not the model-optimal policy."""
    t, stock = state
    if t == env.horizon - 1:
        return 0
    carry = max(stock - env.MEANS[t], 0.)
    target = env.MEANS[t + 1] + .5
    distance = (carry + env.ACTIONS - target) ** 2
    return int(np.argmin(np.where(mask, distance, np.inf)))


def cafe_episode(env, policy, seed):
    state, info = env.reset(seed=int(seed))
    rows, total = [], 0.
    for t in range(env.horizon):
        action = policy(state, info['action_mask'])
        nxt, reward, done, truncated, info = env.step(action)
        rows.append({'hour': 10 + t, 'action': action, 'reward': reward, **info})
        total += reward
        state = nxt
        if done or truncated:
            break
    return total, rows


def cafe_evaluate(agent, seeds, env_kwargs=None):
    config = getattr(agent, 'env_kwargs', {}) if env_kwargs is None else env_kwargs
    env = CookieCafe(**config)
    records = []
    for seed in seeds:
        policy = (lambda s, m: cafe_rule(s, m, env)) if agent is None else (lambda s, m: agent.act(s, m))
        ret, rows = cafe_episode(env, policy, seed)
        records.append(dict(return_=ret, profit=sum(r['cash_profit'] for r in rows),
                            waste=sum(r['waste'] for r in rows), missed=sum(r['missed'] for r in rows)))
    env.close()
    return pd.DataFrame(records)


def train_cafe(rules, episodes=1200, seed=0, log_every=100, env_kwargs=None):
    env = CookieCafe(**(env_kwargs or {}))
    agent = CafeQAgent(env, rules, seed)
    schedule_rng = np.random.default_rng(seed + 1000)
    history = []
    started = time.perf_counter()
    for episode in range(episodes):
        epsilon = np.interp(episode, [0, episodes * .75], [.8, .05])
        state, info = env.reset(seed=int(schedule_rng.integers(2**31)))
        total = 0.
        for _ in range(env.horizon):
            action = agent.act(state, info['action_mask'], epsilon)
            nxt, reward, done, truncated, info = env.step(action)
            agent.update(state, action, reward, nxt, done, info['action_mask'])
            total += reward
            state = nxt
            if done or truncated:
                break
        history.append({'episode': episode + 1, 'return': total, 'epsilon': epsilon})
    env.close()
    print(f'Café: {episodes:,} days, {episodes * env.horizon:,} transitions, '
          f'{time.perf_counter() - started:.1f}s; {np.count_nonzero(agent.visits):,} visited states.')
    return agent, pd.DataFrame(history)


def plot_cafe(agent, history, example_seed=2026, save_dir=None, reference=None):
    """Start with one real day; compare recommendations at the SAME observed states."""
    from .cafe_reference import solve_cafe_reference
    env = CookieCafe(**agent.env_kwargs)
    reference = solve_cafe_reference(env) if reference is None else reference
    _, rows = cafe_episode(env, lambda s, m: agent.act(s, m), example_seed)
    hours = np.array([r['hour'] for r in rows])
    demand = np.array([r['demand'] for r in rows])
    stock = np.array([r['inventory'] for r in rows])
    orders = np.array([r['produced'] for r in rows])
    model_orders = []
    for row in rows:
        env.t, env.stock = row['hour'] - 10, row['inventory']
        model_orders.append(env.ACTIONS[reference.act(env.observation(), env.action_mask())])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    window = min(100, len(history))
    axes[0, 0].plot(history.episode, history['return'].rolling(window, min_periods=1).mean(), color='#454545')
    axes[0, 0].set(title=f'Training reward · rolling {window}-day mean', xlabel='Training days (exploration on)',
                   ylabel='Reward (credits / day)')
    axes[0, 1].plot(hours, env.MEANS, 'o-', color=COOKIE_BROWN, lw=2.5, label='Known mean demand')
    axes[0, 1].bar(hours, demand, width=.45, facecolor='none', edgecolor=COOKIE_BROWN,
                   hatch='///', label=f'Actual demand · day {example_seed}')
    axes[0, 1].set(title='Random demand · lunch and evening peaks',
                   xlabel='Hour', ylabel='Customers', xticks=hours)
    axes[0, 1].legend(frameon=False, fontsize=9)
    axes[1, 0].step(hours, stock, where='mid', color=COOKIE_BROWN, lw=2, label='Stock before customers arrive')
    axes[1, 0].bar(hours, [r['sales'] for r in rows], color=COOKIE_BROWN, alpha=.35,
                   width=.45, label='Actual sales')
    axes[1, 0].set(title=f'A real café day · missed {sum(r["missed"] for r in rows)}, closing waste {rows[-1]["waste"]}',
                   xlabel='Hour', ylabel='Cookies', xticks=hours)
    # Reserve space below the axes so the legend never covers stock or sales.
    axes[1, 0].legend(loc='upper left', bbox_to_anchor=(0., -.23), ncol=2,
                      frameon=False, fontsize=9, borderaxespad=0.)
    axes[1, 1].plot(hours, orders, 'o-', color=COOKIE_BROWN, lw=2,
                    label='Learned: prepare now')
    axes[1, 1].plot(hours, model_orders, 's--', color='#4d3525', lw=1.8,
                    label='Model-optimal: same hour and stock')
    axes[1, 1].set(title='What to prepare at each observed state', xlabel='Decision hour (ready next hour)',
                   ylabel='Cookies to prepare', xticks=hours, yticks=[0, 2, 4], ylim=(-.3, 4.5))
    axes[1, 1].legend(loc='upper left', bbox_to_anchor=(0., -.23),
                      frameon=False, fontsize=9, borderaxespad=0.)
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.15)
    fig.suptitle(f'{env.PRODUCT_NAME} café · only demand is uncertain', fontsize=16, weight='bold')
    if save_dir is not None:
        fig.savefig(Path(save_dir) / 'cafe_results.png', dpi=180)
    plt.show()
    env.close()
    return rows


def plot_cafe_policy(agent, save_dir=None, reference=None):
    """Full hour-by-stock policy table: no ingredient or other-product slice."""
    from .cafe_reference import solve_cafe_reference
    env = CookieCafe(**agent.env_kwargs)
    reference = solve_cafe_reference(env, gamma=agent.gamma) if reference is None else reference
    if reference.env_kwargs != agent.env_kwargs or reference.gamma != agent.gamma:
        raise ValueError('Use a reference with the same game parameters and discount as the agent.')
    shape = (env.stock_cap + 1, env.horizon)
    model, learned = np.zeros(shape), np.full(shape, np.nan)
    records = []
    for stock in range(env.stock_cap + 1):
        for t in range(env.horizon):
            env.t, env.stock = t, stock
            mask, key = env.action_mask(), (t, stock)
            optimal_action = reference.act(key, mask)
            model[stock, t] = env.ACTIONS[optimal_action]
            visited = bool(agent.visits[key])
            action = int(np.argmax(np.where(mask, agent.q[key], -np.inf))) if visited else None
            if visited:
                learned[stock, t] = env.ACTIONS[action]
            records.append(dict(hour=10 + t, stock=stock, visits=int(agent.visits[key]),
                                reachable_from_default_opening=bool(reference.reachable[key]),
                                model_action=optimal_action, model_prepare=int(env.ACTIONS[optimal_action]),
                                learned_action=action, learned_prepare=int(env.ACTIONS[action]) if visited else None,
                                optimal_action_value_gap=(reference.values[key] - reference.q[(*key, action)])
                                if visited else np.nan))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained', sharex=True, sharey=True)
    cmap = ListedColormap(['#fbf1e7', '#d5a57a', COOKIE_BROWN])
    cmap.set_bad('white')
    norm = BoundaryNorm([-1, 1, 3, 5], cmap.N)
    for column, (ax, data) in enumerate(zip(axes, [model, learned])):
        im = ax.imshow(data, origin='lower', aspect='auto', cmap=cmap, norm=norm)
        for stock in range(env.stock_cap + 1):
            for t in range(env.horizon):
                value = data[stock, t]
                ax.text(t, stock, '?' if np.isnan(value) else str(int(value)), ha='center', va='center',
                        color='white' if value == 4 else '#392b21', fontsize=10)
        ax.set(xticks=range(env.horizon), xticklabels=range(10, 10 + env.horizon),
               yticks=range(env.stock_cap + 1), xlabel='Decision hour', ylabel='Cookies currently in stock',
               title='Model-optimal · known demand distribution' if column == 0 else 'Learned Q-table · greedy choice')
    fig.colorbar(im, ax=axes, ticks=[0, 2, 4], shrink=.85, label='Cookies to prepare now')
    fig.suptitle('At this hour, with this stock, how many cookies should we prepare?', fontsize=14, weight='bold')
    fig.supxlabel('Numbers = portions to prepare.  ? = no training update for that state; it does not mean actions are forbidden.',
                  fontsize=10)
    if save_dir is not None:
        fig.savefig(Path(save_dir) / 'cafe_policy.png', dpi=180)
    plt.show()
    env.close()
    return pd.DataFrame(records)
