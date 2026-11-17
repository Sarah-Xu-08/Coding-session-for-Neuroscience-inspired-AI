"""Provided known-model reference for the one-product, demand-only cookie game."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import poisson
from .cafe import CookieCafe, cafe_model_parameters


@dataclass
class CafeReference:
    q: np.ndarray
    values: np.ndarray
    reachable: np.ndarray
    transitions: np.ndarray
    gamma: float
    env_kwargs: dict

    def act(self, state, mask):
        valid = np.flatnonzero(mask)
        return int(valid[np.argmax(self.q[tuple(state)][valid])])

    @property
    def expected_opening_return(self):
        return float(self.values[0, self.env_kwargs['opening_stock']])


def solve_cafe_reference(env=None, gamma=1.):
    """Backward induction over hour and stock; integrate the full Poisson tail.

    No future customers are revealed. If stock is k, leftover j>0 means demand
    k-j; leftover zero includes ALL demands >=k. No truncated-demand approximation.
    """
    env = CookieCafe() if env is None else env
    cap, horizon = env.stock_cap, env.horizon
    stocks = np.arange(cap + 1)
    kernels = np.zeros((horizon, cap + 1, cap + 1))
    support = np.zeros_like(kernels, dtype=bool)
    for t, mean in enumerate(env.MEANS):
        for stock in stocks:
            kernels[t, stock, 0] = poisson.sf(stock - 1, mean)
            leftovers = np.arange(1, stock + 1)
            kernels[t, stock, leftovers] = poisson.pmf(stock - leftovers, mean)
            support[t, stock, :stock + 1] = mean > 0
            support[t, stock, stock] = True
    values = np.zeros((horizon + 1, cap + 1))
    q = np.full((*values.shape, len(env.ACTIONS)), -np.inf)
    q[horizon] = 0.
    for t in range(horizon - 1, -1, -1):
        for stock in stocks:
            p = kernels[t, stock]
            leftover_mean = float(p @ stocks)
            sold_mean = stock - leftover_mean
            base = sold_mean * env.price - env.missed_penalty * (env.MEANS[t] - sold_mean)
            if t == 0:
                base -= env.opening_stock * env.cost
            if t == horizon - 1:
                base -= leftover_mean * env.disposal_fee
            for action, order in enumerate(env.ACTIONS):
                if stock + order > cap or (t == horizon - 1 and action != 0):
                    continue
                future = p[:stock + 1] @ values[t + 1, order:order + stock + 1]
                q[t, stock, action] = base - order * env.cost + gamma * future
            values[t, stock] = q[t, stock].max()
    reachable = np.zeros_like(values, dtype=bool)
    reachable[0, env.opening_stock] = True
    for t in range(horizon):
        for stock in np.flatnonzero(reachable[t]):
            for action, order in enumerate(env.ACTIONS):
                if stock + order > cap or (t == horizon - 1 and action != 0):
                    continue
                reachable[t + 1, np.flatnonzero(support[t, stock]) + order] = True
    return CafeReference(q, values, reachable, kernels, gamma, cafe_model_parameters(env))
