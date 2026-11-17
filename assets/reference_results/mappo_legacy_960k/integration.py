"""Simple Spread integration for skrl 1.4.3 MAPPO (no local PPO update).

Only model architecture, environment batching, and experiment I/O live here.
skrl owns action sampling, rollout memory, advantages, PPO losses and optimization.
Actors receive their own 18-value observations. Training critics receive the
environment's 54-value global state. Separate actors/critics are used per agent.
"""
import copy
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from skrl.memories.torch import RandomMemory
from skrl.models.torch import CategoricalMixin, DeterministicMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler

from . import ROOT
from .spread import make_spread, reset_spread, spread_rollout

REFERENCE = ROOT / 'assets/reference_results/multi_agent'
NAMES = ['agent_0', 'agent_1', 'agent_2']
SETTINGS = dict(rollouts=120, learning_epochs=4, mini_batches=4,
                learning_rate=3e-4, discount_factor=.99, **{'lambda': .95},
                entropy_loss_scale=.01, grad_norm_clip=.5, ratio_clip=.2,
                value_loss_scale=1., time_limit_bootstrap=False)


def _network(inputs, outputs):
    net = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(),
                        nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, outputs))
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(net[-1].weight, gain=.01 if outputs == 5 else 1.)
    return net


class SpreadActor(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space):
        Model.__init__(self, observation_space, action_space, device='cpu')
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        self.net = _network(self.num_observations, self.num_actions)

    def compute(self, inputs, role):
        return self.net(inputs['states']), {}


class SpreadCritic(DeterministicMixin, Model):
    def __init__(self, state_space, action_space):
        Model.__init__(self, state_space, action_space, device='cpu')
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = _network(self.num_observations, 1)

    def compute(self, inputs, role):
        return self.net(inputs['states']), {}


def build_mappo(seed=0, num_envs=1, training=False):
    """Construct the library MAPPO object; only training needs rollout memories."""
    torch.manual_seed(seed)
    env = make_spread()
    obs = {n: env.observation_space(n) for n in NAMES}
    actions = {n: env.action_space(n) for n in NAMES}
    shared = {n: env.state_space for n in NAMES}
    env.close()
    models = {n: dict(policy=SpreadActor(obs[n], actions[n]),
                      value=SpreadCritic(shared[n], actions[n])) for n in NAMES}
    memories = {n: RandomMemory(memory_size=SETTINGS['rollouts'], num_envs=num_envs,
                                 device='cpu') for n in NAMES} if training else None
    cfg = copy.deepcopy(MAPPO_DEFAULT_CONFIG)
    cfg.update(SETTINGS)
    cfg.update(value_preprocessor=RunningStandardScaler,
               value_preprocessor_kwargs=dict(size=1, device='cpu'))
    cfg['experiment'].update(write_interval=0, checkpoint_interval=0, wandb=False)
    agent = MAPPO(possible_agents=NAMES, models=models, memories=memories,
                  observation_spaces=obs, shared_observation_spaces=shared,
                  action_spaces=actions, device='cpu', cfg=cfg)
    agent.init()
    agent.set_running_mode('train' if training else 'eval')
    return agent


class MAPPOPolicy:
    """Greedy categorical evaluation: local observation in, one action out."""
    def __init__(self, agent):
        self.agent = agent

    @torch.no_grad()
    def act(self, name, observation):
        state = torch.as_tensor(np.asarray(observation), dtype=torch.float32).reshape(1, 18)
        logits, _ = self.agent.policies[name].compute({'states': state}, role='policy')
        return int(logits.argmax(dim=-1).item())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_mappo(folder=REFERENCE):
    """Load the supplied genuine library checkpoint after checking provenance."""
    folder = Path(folder)
    metadata = json.loads((folder / 'mappo_manifest.json').read_text())
    if metadata['library_version'] != version('skrl'):
        raise ValueError('Use the pinned skrl version from environment.yml.')
    for name, digest in metadata['files_sha256'].items():
        if sha256(folder / name) != digest:
            raise ValueError(f'MAPPO reference file changed: {name}')
    agent = build_mappo()
    agent.load(str(folder / 'mappo.pt'))
    agent.set_running_mode('eval')
    agent.set_mode('eval')
    return MAPPOPolicy(agent), metadata


def evaluate_mappo(policy, seeds, layout='random'):
    records = []
    for seed in seeds:
        rollout = spread_rollout('mappo', seed=int(seed), layout=layout, policy=policy)
        records.append(dict(seed=seed, layout=layout, controller='MAPPO (skrl)',
                            team_return=rollout['history'].team_reward.sum(),
                            final_reward=rollout['rewards'][-1]))
    return pd.DataFrame(records)


def train_mappo(folder, joint_steps=960000, num_envs=16, seed=0, layout='random'):
    """Offline training runner. All learning calls go to the installed library.

    One joint step is one simultaneous decision by all three agents in one
    environment. Batching collects num_envs such steps per runner iteration.
    We select on fixed development seeds; test seeds are evaluated only after.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    agent = build_mappo(seed, num_envs, training=True)
    envs = [make_spread() for _ in range(num_envs)]
    resets = np.zeros(num_envs, dtype=int)
    observations = [reset_spread(e, seed=seed * 100000 + i, layout=layout)[0]
                    for i, e in enumerate(envs)]
    iterations = int(np.ceil(joint_steps / (num_envs * SETTINGS['rollouts']))
                     * SETTINGS['rollouts'])
    best, logs, returns = -np.inf, [], np.zeros(num_envs)
    episode_returns = []
    started = time.perf_counter()
    dev_seeds = list(range(71000, 71008))
    test_seeds = list(range(91000, 91020))
    next_check = 0
    try:
        for t in range(iterations):
            states = {n: torch.tensor(np.stack([o[n] for o in observations]), dtype=torch.float32)
                      for n in NAMES}
            shared = torch.tensor(np.stack([e.state() for e in envs]), dtype=torch.float32)
            agent.pre_interaction(t, iterations)
            with torch.no_grad():
                actions, _, _ = agent.act(states, timestep=t, timesteps=iterations)
            outputs = [e.step({n: int(actions[n][i].item()) for n in NAMES})
                       for i, e in enumerate(envs)]
            next_observations = [out[0] for out in outputs]
            next_states = {n: torch.tensor(np.stack([o[n] for o in next_observations]), dtype=torch.float32)
                           for n in NAMES}
            rewards = {n: torch.tensor([[out[1][n]] for out in outputs], dtype=torch.float32) for n in NAMES}
            terminated = {n: torch.tensor([[out[2][n]] for out in outputs]) for n in NAMES}
            truncated = {n: torch.tensor([[out[3][n]] for out in outputs]) for n in NAMES}
            shared_next = torch.tensor(np.stack([e.state() for e in envs]), dtype=torch.float32)
            with torch.no_grad():
                agent.record_transition(states, actions, rewards, next_states, terminated, truncated,
                                        dict(shared_states=shared, shared_next_states=shared_next),
                                        timestep=t, timesteps=iterations)
            agent.post_interaction(t, iterations)
            returns += rewards[NAMES[0]].numpy().ravel()  # Count the shared score ONCE.
            observations = next_observations
            for i, out in enumerate(outputs):
                if all(out[2][n] or out[3][n] for n in NAMES):
                    episode_returns.append(float(returns[i]))
                    returns[i] = 0
                    resets[i] += 1
                    observations[i] = reset_spread(envs[i], seed=seed * 100000 + i + int(resets[i]) * num_envs,
                                                   layout=layout)[0]
            if (t + 1) % SETTINGS['rollouts'] == 0 and ((t + 1) * num_envs >= next_check or t + 1 == iterations):
                dev = evaluate_mappo(MAPPOPolicy(agent), dev_seeds, layout=layout)
                score = float(dev.team_return.mean())
                logs.append(dict(joint_steps=(t + 1) * num_envs,
                                 mean_last_100_training_returns=float(np.mean(episode_returns[-100:])),
                                 development_return=score))
                if score > best:
                    best = score
                    agent.save(str(folder / 'mappo.pt'))
                print(f"MAPPO {(t+1)*num_envs} joint steps | development return {score:.2f} | {time.perf_counter()-started:.1f}s", flush=True)
                next_check = (t + 1) * num_envs + 48000
                # No writer is enabled; avoid retaining unused library diagnostic logs.
                agent.tracking_data.clear()
        pd.DataFrame(logs).to_csv(folder / 'mappo_training.csv', index=False)
        agent.load(str(folder / 'mappo.pt'))
        policy = MAPPOPolicy(agent)
        records = [evaluate_mappo(policy, test_seeds, layout=layout)]
        for strategy in ['nearest', 'assignment']:
            rows = []
            for s in test_seeds:
                r = spread_rollout(strategy, seed=s, layout=layout)
                rows.append(dict(seed=s, layout=layout, controller=strategy,
                                 team_return=r['history'].team_reward.sum(), final_reward=r['rewards'][-1]))
            records.append(pd.DataFrame(rows))
        pd.concat(records, ignore_index=True).to_csv(folder / 'spread_evaluation.csv', index=False)
        metadata = dict(algorithm='MAPPO', library='skrl', library_version=version('skrl'),
                        library_source='https://github.com/Toni-SM/skrl/tree/1.4.3',
                        training_seed=seed, joint_steps=iterations * num_envs,
                        individual_actions=iterations * num_envs * 3, num_envs=num_envs,
                        settings=SETTINGS, actor='18 local observation values -> 64 tanh -> 64 tanh -> 5 logits',
                        critic='54 global state values -> 64 tanh -> 64 tanh -> 1 value, per agent',
                        parameter_sharing=False, local_ratio=0., n_agents=3, max_cycles=60,
                        training_layout=layout, development_seeds=dev_seeds, test_seeds=test_seeds,
                        selection='best mean development team return; greedy categorical evaluation',
                        reward='negative sum of each landmark distance to its nearest agent; no shaping',
                        demo=dict(seed=2026, layout='clustered', selection='fixed teaching scene, not a generalization estimate'),
                        versions={n: version(n) for n in ['torch','numpy','skrl','mpe2','pettingzoo']},
                        seconds=round(time.perf_counter()-started, 2),
                        files_sha256={name: sha256(folder/name) for name in
                                      ['mappo.pt','mappo_training.csv','spread_evaluation.csv']})
        (folder / 'mappo_manifest.json').write_text(json.dumps(metadata, indent=2))
        return metadata
    finally:
        for env in envs:
            env.close()
