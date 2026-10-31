"""Provided synthetic cafe, tabular storage, evaluation and visualizations."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
COLORS = {'Q-learning':'#2563a6', 'DQN':'#db7130', 'Rule':'#438c68'}
#@title Provided café simulator · inspect the state, reward, and action mask
class DessertCafe(gym.Env):
    """Small finite-horizon MDP. All money values are synthetic credits.

    State = (time, cookie_stock, cake_stock, remaining_pistachio_units).
    One portion of either dessert uses one simplified ingredient unit.
    Prepare now; the portions become available at the next decision time.
    Both desserts are discarded at closing under this teaching assumption.
    """
    metadata = {'render_modes': []}
    ACTIONS = np.array([(0,0), (2,0), (0,2), (4,0), (2,2), (0,4)], dtype=int)
    NAMES = ['Dubai cookie', 'Pistachio cake']
    MEANS = np.array([[.3,.5,.7,1.,1.7,3.,4.,3.5],
                      [.5,1.,3.5,4.,2.,1.,.5,.5]])

    def __init__(self, stock_cap=8, ingredient_budget=24,
                 missed_penalty=0.5, disposal_fee=1.0, substitution=0.25):
        self.stock_cap = stock_cap
        self.ingredient_budget = ingredient_budget
        self.horizon = 8
        self.missed_penalty = missed_penalty
        self.disposal_fee = disposal_fee
        self.substitution = substitution
        self.price = np.array([6.,7.])
        self.cost = np.array([3.,4.])
        self.observation_space = spaces.MultiDiscrete([9, stock_cap+1, stock_cap+1,
                                                       ingredient_budget+1])
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.t = 0
        self.stock = np.array([2,2])
        self.ingredients = ingredient_budget

    def observation(self):
        return np.array([self.t, *self.stock, self.ingredients], dtype=np.int64)

    def action_mask(self):
        if self.t >= self.horizon-1:
            return np.array([True, False, False, False, False, False])
        return ((self.ACTIONS.sum(axis=1) <= self.ingredients)
                & np.all(self.ACTIONS + self.stock <= self.stock_cap, axis=1))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.stock = np.array([2,2], dtype=int)
        self.ingredients = self.ingredient_budget
        self.traffic = []
        # Generate all exogenous arrivals first. Policies evaluated on the same
        # seed therefore face the SAME customer requests and substitution draws.
        for t in range(self.horizon):
            counts = self.np_random.poisson(self.MEANS[:,t])
            requests = np.repeat([0,1], counts)
            self.np_random.shuffle(requests)
            self.traffic.append((requests, self.np_random.random(len(requests))))
        return self.observation(), {'action_mask': self.action_mask()}

    def step(self, action):
        if self.t >= self.horizon:
            raise RuntimeError('The café is closed; call reset().')
        if not self.action_space.contains(action) or not self.action_mask()[action]:
            raise ValueError(f'Action {action} is infeasible in state {self.observation()}.')
        old_stock = self.stock.copy()
        order = self.ACTIONS[action].copy()
        requests, switches = self.traffic[self.t]
        demand = np.bincount(requests, minlength=2)
        sold, missed = np.zeros(2, int), np.zeros(2, int)
        substituted = 0
        for preferred, switch in zip(requests, switches):
            chosen = int(preferred)
            if self.stock[chosen] == 0:
                alternative = 1-chosen
                if switch < self.substitution and self.stock[alternative] > 0:
                    chosen = alternative
                    substituted += 1
                else:
                    missed[preferred] += 1
                    continue
            self.stock[chosen] -= 1
            sold[chosen] += 1
        self.ingredients -= int(order.sum())
        self.stock += order
        opening_cost = float(np.array([2,2]) @ self.cost) if self.t == 0 else 0.
        prep_cost = float(order @ self.cost) + opening_cost
        revenue = float(sold @ self.price)
        self.t += 1
        terminated = self.t == self.horizon
        waste = self.stock.copy() if terminated else np.zeros(2, int)
        disposal = float(waste.sum() * self.disposal_fee)
        cash_profit = revenue - prep_cost - disposal
        reward = cash_profit - self.missed_penalty * int(missed.sum())
        info = dict(action_mask=self.action_mask(), inventory=old_stock,
                    produced=order, demand=demand, sales=sold, missed=missed,
                    waste=waste, substituted=substituted, cash_profit=cash_profit,
                    revenue=revenue, preparation_cost=prep_cost, disposal=disposal,
                    next_inventory=self.stock.copy())
        return self.observation(), float(reward), terminated, False, info

#@title Café learner, forecast-based rule, and training helpers
class CafeQAgent:
    def __init__(self, env, rules, seed=0, alpha=0.15, gamma=1.0):
        self.rules = rules
        self.q = np.zeros((*env.observation_space.nvec, env.action_space.n), np.float32)
        self.visits = np.zeros(tuple(env.observation_space.nvec), np.int32)
        self.rng = np.random.default_rng(seed)
        self.alpha, self.gamma = alpha, gamma

    def act(self, state, mask, epsilon=0.):
        return self.rules.epsilon_greedy(self.q[tuple(state)], epsilon, self.rng, mask)

    def update(self, state, action, reward, next_state, terminated, next_mask):
        key = tuple(state)
        target = self.rules.td_target(reward, terminated, self.q[tuple(next_state)], next_mask, self.gamma)
        self.q[(*key, action)] = self.rules.q_update(self.q[(*key, action)], target, self.alpha)
        self.visits[key] += 1

def cafe_rule(state, mask, env):
    t, cookies, cakes, ingredients = state
    if t == env.horizon-1:
        return 0
    carry = np.maximum(np.array([cookies,cakes]) - env.MEANS[:,t], 0.)
    target = env.MEANS[:,t+1] + 0.5
    # The rule knows the teaching demand pattern; it is a meaningful baseline.
    costs = np.sum((carry[None,:] + env.ACTIONS - target[None,:])**2, axis=1)
    return int(np.argmin(np.where(mask, costs, np.inf)))

def cafe_episode(env, policy, seed):
    state, info = env.reset(seed=int(seed))
    rows, total = [], 0.
    for t in range(env.horizon):
        action = policy(state, info['action_mask'])
        nxt, reward, done, truncated, info = env.step(action)
        rows.append({'hour':10+t, 'action':action, 'reward':reward, **info})
        total += reward
        state = nxt
        if done or truncated:
            break
    return total, rows

def cafe_evaluate(agent, seeds, env_kwargs=None):
    env = DessertCafe(**(env_kwargs or {}))
    records = []
    for seed in seeds:
        policy = (lambda s,m: agent.act(s,m)) if isinstance(agent,CafeQAgent) else (lambda s,m: cafe_rule(s,m,env))
        ret, rows = cafe_episode(env, policy, seed)
        records.append(dict(return_=ret, profit=sum(r['cash_profit'] for r in rows),
                            waste=sum(int(r['waste'].sum()) for r in rows),
                            missed=sum(int(r['missed'].sum()) for r in rows)))
    env.close()
    return pd.DataFrame(records)

def train_cafe(rules, episodes=1200, seed=0, log_every=100, env_kwargs=None):
    env = DessertCafe(**(env_kwargs or {}))
    agent = CafeQAgent(env,rules,seed)
    schedule_rng = np.random.default_rng(seed+1000)
    history = []
    started = time.perf_counter()
    for episode in range(episodes):
        epsilon = np.interp(episode,[0,episodes*.75],[.8,.05])
        state, info = env.reset(seed=int(schedule_rng.integers(2**31)))
        total = 0.
        for _ in range(env.horizon):
            action = agent.act(state,info['action_mask'],epsilon)
            nxt,reward,done,truncated,info = env.step(action)
            agent.update(state,action,reward,nxt,done,info['action_mask'])
            total += reward
            state = nxt
            if done or truncated: break
        history.append({'episode':episode+1,'return':total,'epsilon':epsilon})
    env.close()
    print(f'Café: {episodes:,} days, {episodes*8:,} training transitions, '
          f'{time.perf_counter()-started:.1f}s; {np.count_nonzero(agent.visits):,} visited states.')
    return agent,pd.DataFrame(history)

#@title Café visualization helpers
def plot_cafe(agent, history, example_seed=2026, save_dir=None):
    env = DessertCafe()
    _, rows = cafe_episode(env,lambda s,m:agent.act(s,m),example_seed)
    fig, axes = plt.subplots(2,2,figsize=(12,7),layout='constrained')
    smooth = history['return'].rolling(min(100,len(history)),min_periods=1).mean()
    axes[0,0].plot(history.episode,smooth,color=COLORS['Q-learning'])
    axes[0,0].set(xlabel='Training days',ylabel='Exploratory return (credits/day)',
                  title='Training return · rolling mean')
    hours = np.array([r['hour'] for r in rows])
    for item,ax in enumerate([axes[0,1],axes[1,0]]):
        demand = np.array([r['demand'][item] for r in rows])
        sales = np.array([r['sales'][item] for r in rows])
        order = np.array([r['produced'][item] for r in rows])
        ax.bar(hours-.23,demand,width=.23,label='First-choice demand',alpha=.4)
        ax.bar(hours,sales,width=.23,label='Actual sales')
        ax.plot(hours+.23,order,'o--',label='Prepare for next slot')
        ax.set(xlabel='Decision hour',ylabel='Portions',title=env.NAMES[item])
        ax.legend(fontsize=8)
    ax = axes[1,1]
    waste = np.sum([r['waste'] for r in rows],axis=0)
    missed = np.sum([r['missed'] for r in rows],axis=0)
    ax.bar(np.arange(2)-.17,waste,width=.34,label='Closing waste')
    ax.bar(np.arange(2)+.17,missed,width=.34,label='Missed first-choice requests')
    ax.set_xticks([0,1],['Dubai cookie','Pistachio cake'])
    ax.set(ylabel='Portions',title=f'One held-out day · seed {example_seed}')
    ax.legend(fontsize=8)
    if save_dir is not None:
        fig.savefig(Path(save_dir) / 'cafe_results.png', dpi=180)
    plt.show()
    env.close()
    return rows

def plot_cafe_policy(agent, ingredients=12, other_stock=2, save_dir=None):
    env = DessertCafe()
    fig,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for item,ax in enumerate(axes):
        actions = np.full((9,8),np.nan)
        for stock in range(9):
            for t in range(8):
                pair = [other_stock,other_stock];pair[item]=stock
                state = np.array([t,*pair,ingredients])
                if agent.visits[tuple(state)] == 0:continue
                env.t=t;env.stock=np.array(pair);env.ingredients=ingredients
                mask=env.action_mask()
                values=np.where(mask,agent.q[tuple(state)],-np.inf)
                a=int(np.argmax(values))
                actions[stock,t]=env.ACTIONS[a,item]
        cmap=plt.colormaps['Blues'].copy();cmap.set_bad('#e6e6e6')
        im=ax.imshow(actions,origin='lower',aspect='auto',vmin=0,vmax=4,cmap=cmap)
        ax.set_xticks(range(8),range(10,18));ax.set_yticks(range(9))
        ax.set(xlabel='Decision hour',ylabel='Current stock',title=env.NAMES[item])
        fig.colorbar(im,ax=ax,label='Portions prepared',ticks=[0,2,4])
    fig.suptitle(f'Policy slices: other stock={other_stock}, ingredients={ingredients}; gray = unvisited')
    if save_dir is not None:
        fig.savefig(Path(save_dir) / 'cafe_policy.png', dpi=180)
    plt.show();env.close()
