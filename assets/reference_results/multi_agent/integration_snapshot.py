"""Simple Spread integration for skrl 1.4.3 MAPPO (no local PPO update).

Only model architecture, environment batching, and experiment I/O live here.
skrl owns action sampling, rollout memory, advantages, PPO losses and optimization.
Actors receive their own 18-value observations. Training critics receive the
environment's 54-value global state. Homogeneous agents share one actor/critic.
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
GROUP = 'shared_policy'
LEGACY_SETTINGS = dict(rollouts=120, learning_epochs=4, mini_batches=4,
                      learning_rate=3e-4, discount_factor=.99, **{'lambda': .95},
                      entropy_loss_scale=.01, grad_norm_clip=.5, ratio_clip=.2,
                      value_loss_scale=1., time_limit_bootstrap=False)
SHARED_V2_SETTINGS = dict(rollouts=60, learning_epochs=10, mini_batches=1,
                learning_rate=5e-4, discount_factor=.99, **{'lambda': .95},
                entropy_loss_scale=.01, grad_norm_clip=10., ratio_clip=.2,
                clip_predicted_values=True, value_clip=.2,
                value_loss_scale=1., time_limit_bootstrap=False)
SETTINGS = dict(rollouts=60, learning_epochs=10, mini_batches=1,
                learning_rate=3e-4, discount_factor=.99, **{'lambda': .95},
                entropy_loss_scale=.02, grad_norm_clip=.5, ratio_clip=.2,
                clip_predicted_values=False, value_clip=1000000.,
                value_loss_scale=1., time_limit_bootstrap=True)
REWARD_SCALE = .1
DEV_SEEDS = list(range(171000, 171016))
TEST_SEEDS = list(range(271000, 271050))


def _network(inputs, outputs, legacy=False):
    if legacy:
        net = nn.Sequential(nn.Linear(inputs, 64), nn.Tanh(),
                            nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, outputs))
    else:
        # The normalization/architecture follows the official MAPPO MLP pattern.
        net = nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs, 64), nn.ReLU(), nn.LayerNorm(64),
                            nn.Linear(64, 64), nn.ReLU(), nn.LayerNorm(64), nn.Linear(64, outputs))
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(net[-1].weight, gain=.01 if outputs == 5 else 1.)
    return net


class SpreadActor(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, legacy=False, geometric=False):
        Model.__init__(self, observation_space, action_space, device='cpu')
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        self.geometric = geometric
        self.net = _network(24 if geometric else self.num_observations, self.num_actions, legacy)

    def compute(self, inputs, role):
        states = local_distance_features(inputs['states']) if self.geometric else inputs['states']
        return self.net(states), {}


class SpreadCritic(DeterministicMixin, Model):
    def __init__(self, state_space, action_space, legacy=False, geometric=False):
        Model.__init__(self, state_space, action_space, device='cpu')
        DeterministicMixin.__init__(self, clip_actions=False)
        self.geometric = geometric
        self.net = _network(72 if geometric else self.num_observations, 1, legacy)

    def compute(self, inputs, role):
        states = inputs['states']
        if self.geometric:
            states = local_distance_features(states.reshape(-1,18)).reshape(states.shape[0],72)
        return self.net(states), {}


def local_distance_features(observations):
    """Describe geometry already present in a local observation; never choose actions.

    Append own distance to each landmark and nearest teammate distance to each
    landmark. No global state, target assignment, expert action, or reward bonus
    is supplied to an actor. The library still learns every action probability.
    """
    landmarks = observations[:,4:10].reshape(-1,3,2)
    teammates = observations[:,10:14].reshape(-1,2,2)
    own_distances = torch.linalg.vector_norm(landmarks,dim=-1)
    teammate_distances = torch.linalg.vector_norm(landmarks[:,:,None,:]-teammates[:,None,:,:],dim=-1).amin(dim=-1)
    return torch.cat([observations,own_distances,teammate_distances],dim=-1)


def build_mappo(seed=0, num_envs=1, training=False, legacy=False, shared_v2=False, geometric=True):
    """Construct the library MAPPO object; only training needs rollout memories."""
    torch.manual_seed(seed)
    env = make_spread()
    names = NAMES if legacy else [GROUP]
    settings = LEGACY_SETTINGS if legacy else SHARED_V2_SETTINGS if shared_v2 else SETTINGS
    obs = {n: env.observation_space(NAMES[0]) for n in names}
    actions = {n: env.action_space(NAMES[0]) for n in names}
    shared = {n: env.state_space for n in names}
    env.close()
    geometric = geometric and not legacy and not shared_v2
    models = {n: dict(policy=SpreadActor(obs[n], actions[n], legacy, geometric),
                      value=SpreadCritic(shared[n], actions[n], legacy, geometric)) for n in names}
    memories = {n: RandomMemory(memory_size=settings['rollouts'], num_envs=num_envs if legacy else num_envs*3,
                                 device='cpu') for n in names} if training else None
    cfg = copy.deepcopy(MAPPO_DEFAULT_CONFIG)
    cfg.update(settings)
    if legacy or shared_v2:
        cfg.update(value_preprocessor=RunningStandardScaler,
                   value_preprocessor_kwargs=dict(size=1, device='cpu'))
    else:
        # Positive constant scaling preserves the task objective. Scores are
        # always reported in the original, unscaled environment reward units.
        cfg['rewards_shaper'] = lambda rewards, timestep, timesteps: REWARD_SCALE * rewards
        # skrl 1.4.3 tests the per-agent clipping dictionary's truthiness. A wide
        # value_clip makes that branch an identity over the critic's value range,
        # so the requested unclipped critic actually remains unclipped.
    cfg['experiment'].update(write_interval=0, checkpoint_interval=0, wandb=False)
    agent = MAPPO(possible_agents=names, models=models, memories=memories,
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
        key = GROUP if GROUP in self.agent.policies else name
        logits, _ = self.agent.policies[key].compute({'states': state}, role='policy')
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
    agent = build_mappo(legacy=not metadata.get('parameter_sharing', False),
                        shared_v2=metadata.get('format_version') == 2,
                        geometric=metadata.get('format_version',1) >= 4)
    agent.load(str(folder / 'mappo.pt'))
    agent.set_running_mode('eval')
    agent.set_mode('eval')
    return MAPPOPolicy(agent), metadata


def evaluate_mappo(policy, seeds, layout='random'):
    records = []
    for seed in seeds:
        rollout = spread_rollout('mappo', seed=int(seed), layout=layout, policy=policy)
        records.append(_score_rollout(rollout, 'MAPPO (skrl)'))
    return pd.DataFrame(records)


def _score_rollout(rollout, controller):
    distances = np.linalg.norm(rollout['positions'][-1, :, None, :] - rollout['landmarks'][None, :, :], axis=-1)
    return dict(seed=rollout['seed'], layout=rollout['layout'], controller=controller,
                team_return=float(rollout['history'].team_reward.sum()),
                final_reward=float(rollout['rewards'][-1]),
                landmarks_within_02=int((distances.min(axis=0) <= .2).sum()),
                max_agent_radius=float(np.linalg.norm(rollout['positions'], axis=-1).max()))


def pack_local_observations(observations):
    """Environment-major, agent-minor batch. Each row contains ONE local view."""
    return torch.tensor(np.stack([obs[name] for obs in observations for name in NAMES]), dtype=torch.float32)


def train_mappo(folder, joint_steps=3000000, num_envs=16, seed=0, layout='mixed',
                resume=False, evaluate_test=False):
    """Train shared-parameter MAPPO with the installed library, outside the video.

    Homogeneous agents form the batch dimension of ONE library policy group.
    Local observation rows and repeated global critic state rows have the same
    environment-major, agent-minor order. One optimizer updates the shared models;
    we do not alias one model across three independent library optimizers.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    agent = build_mappo(seed, num_envs, training=True)
    start = 0
    best = -np.inf
    selected_step = 0
    logs = []
    stages = []
    if resume:
        previous = json.loads((folder/'mappo_manifest.json').read_text())
        if previous.get('format_version') != 4 or previous['settings'] != SETTINGS:
            raise ValueError('Resume requires the same shared MAPPO configuration.')
        if previous.get('test_evaluation_performed'):
            raise ValueError('Final test has already been inspected; use development-only candidates for tuning.')
        agent.load(str(folder/'last.pt'))
        state = torch.load(folder/'runner_state.pt', weights_only=True)
        torch.set_rng_state(state['torch_rng'])
        start = previous['joint_steps']
        best = previous['best_development_return']
        selected_step = previous['selected_joint_steps']
        logs = pd.read_csv(folder/'mappo_training.csv').to_dict('records')
        stages = previous['training_stages']
    if joint_steps <= start:
        raise ValueError('The requested total budget must exceed completed training.')
    rng = np.random.default_rng(np.random.SeedSequence([seed, start, 4781]))
    # Training seeds occupy a disjoint numeric range from development/test seeds.
    def reset_training(env):
        env_seed = int(rng.integers(1000000000, 2000000000))
        chosen = ('clustered' if rng.random() < .2 else 'random') if layout == 'mixed' else layout
        return reset_spread(env, seed=env_seed, layout=chosen)[0]
    envs = [make_spread() for _ in range(num_envs)]
    observations = [reset_training(e) for e in envs]
    iterations = int(np.ceil((joint_steps-start) / (num_envs * SETTINGS['rollouts'])) * SETTINGS['rollouts'])
    returns = np.zeros(num_envs)
    episode_returns = []
    started = time.perf_counter()
    stages.append(dict(start=start, requested_total=joint_steps, layout=layout, seed=seed,
                       environment_reset='fresh environments at the stage boundary'))
    next_check = 0
    plan = dict(settings=SETTINGS, development_seeds=DEV_SEEDS, final_test_seeds=TEST_SEEDS,
                selection='best mean development return; no final test evaluation during training',
                training_stages=stages)
    (folder/'training_plan.json').write_text(json.dumps(plan, indent=2))
    try:
        for t in range(iterations):
            states = {GROUP: pack_local_observations(observations)}
            shared = torch.tensor(np.repeat(np.stack([e.state() for e in envs]), 3, axis=0), dtype=torch.float32)
            agent.pre_interaction(t, iterations)
            with torch.no_grad():
                actions, _, _ = agent.act(states, timestep=t, timesteps=iterations)
            action_array = actions[GROUP].reshape(num_envs, 3)
            outputs = [e.step({n: int(action_array[i,j]) for j,n in enumerate(NAMES)}) for i,e in enumerate(envs)]
            next_observations = [out[0] for out in outputs]
            next_states = {GROUP: pack_local_observations(next_observations)}
            rewards = {GROUP: torch.tensor([[out[1][n]] for out in outputs for n in NAMES], dtype=torch.float32)}
            terminated = {GROUP: torch.tensor([[out[2][n]] for out in outputs for n in NAMES])}
            truncated = {GROUP: torch.tensor([[out[3][n]] for out in outputs for n in NAMES])}
            shared_next = torch.tensor(np.repeat(np.stack([e.state() for e in envs]), 3, axis=0), dtype=torch.float32)
            with torch.no_grad():
                agent.record_transition(states, actions, rewards, next_states, terminated, truncated,
                                        dict(shared_states=shared, shared_next_states=shared_next),
                                        timestep=t, timesteps=iterations)
            agent.post_interaction(t, iterations)
            returns += np.array([out[1][NAMES[0]] for out in outputs])  # Shared score counted once.
            observations = next_observations
            for i, out in enumerate(outputs):
                if all(out[2][n] or out[3][n] for n in NAMES):
                    episode_returns.append(float(returns[i]))
                    returns[i] = 0
                    observations[i] = reset_training(envs[i])
            steps = start + (t+1)*num_envs
            if (t+1) % SETTINGS['rollouts'] == 0 and (steps >= next_check or t+1 == iterations):
                dev = evaluate_mappo(MAPPOPolicy(agent), DEV_SEEDS)
                score = float(dev.team_return.mean())
                diagnostic = {key: float(np.mean(value)) for key,value in agent.tracking_data.items() if value}
                logs.append(dict(joint_steps=steps,
                                 mean_last_100_training_returns=float(np.mean(episode_returns[-100:])),
                                 development_return=score, development_final_reward=float(dev.final_reward.mean()),
                                 development_landmarks_within_02=float(dev.landmarks_within_02.mean()),
                                 development_max_radius=float(dev.max_agent_radius.max()),
                                 **diagnostic))
                if score > best:
                    best, selected_step = score, steps
                    agent.save(str(folder/'mappo.pt'))
                agent.save(str(folder/'last.pt'))
                torch.save(dict(torch_rng=torch.get_rng_state()), folder/'runner_state.pt')
                pd.DataFrame(logs).to_csv(folder/'mappo_training.csv', index=False)
                print(f"MAPPO {steps:,} joint steps | dev return {score:.2f}, final {dev.final_reward.mean():.3f}, "
                      f"landmarks {dev.landmarks_within_02.mean():.2f}/3 | {time.perf_counter()-started:.1f}s", flush=True)
                next_check = steps + 240000
                agent.tracking_data.clear()
        pd.DataFrame(logs).to_csv(folder/'mappo_training.csv', index=False)
        metadata = dict(format_version=4, algorithm='MAPPO', library='skrl', library_version=version('skrl'),
                        library_source='https://github.com/Toni-SM/skrl/tree/1.4.3',
                        architecture_source='https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/utils/mlp.py',
                        training_seed=seed, joint_steps=start+iterations*num_envs,
                        individual_actions=(start+iterations*num_envs)*3, num_envs=num_envs,
                        settings=SETTINGS, actor='18 local observation values -> 24 descriptive distance features; LayerNorm, 64 ReLU LayerNorm, 64 ReLU LayerNorm, 5 logits',
                        critic='54 global state values -> 72 descriptive distance features; LayerNorm, 64 ReLU LayerNorm, 64 ReLU LayerNorm, 1 value',
                        parameter_sharing=True, library_policy_group=GROUP,
                        training_reward_scale=REWARD_SCALE, value_normalization=False,
                        value_clipping='disabled via wide clip window for skrl 1.4.3 compatibility',
                        timeout_handling='library time-limit bootstrap; 60-step cutoff is truncation, not task termination',
                        batch_order='environment-major, agent-minor; actor local rows, critic global rows repeated three times',
                        local_ratio=0., n_agents=3, max_cycles=60,
                        training_layout='80% random, 20% fixed clustered teaching start' if layout=='mixed' else layout,
                        training_seed_range=[1000000000,2000000000], training_stages=stages,
                        development_seeds=DEV_SEEDS, test_seeds=TEST_SEEDS,
                        selected_joint_steps=selected_step, best_development_return=best,
                        test_evaluation_performed=False,
                        selection='best mean development team return; greedy categorical evaluation',
                        reward='negative sum of each landmark distance to its nearest agent; no shaping',
                        demo=dict(seed=2026, layout='clustered', selection='fixed practiced teaching start; not held out'),
                        versions={n: version(n) for n in ['torch','numpy','skrl','mpe2','pettingzoo']},
                        seconds_this_stage=round(time.perf_counter()-started,2),
                        files_sha256={name:sha256(folder/name) for name in ['mappo.pt','mappo_training.csv','training_plan.json']})
        (folder/'mappo_manifest.json').write_text(json.dumps(metadata,indent=2))
        if evaluate_test:
            return finalize_mappo(folder)
        return metadata
    finally:
        for env in envs:
            env.close()


def finalize_mappo(folder=REFERENCE):
    """Evaluate the selected checkpoint once, after development choices are frozen."""
    folder = Path(folder)
    policy, metadata = load_mappo(folder)
    if metadata.get('test_evaluation_performed'):
        raise ValueError('The final test is already recorded. Do not tune against these results.')
    seeds = metadata['test_seeds']
    records = [evaluate_mappo(policy, seeds)]
    for strategy in ['nearest','assignment']:
        records.append(pd.DataFrame([_score_rollout(spread_rollout(strategy, seed=s, layout='random'),strategy)
                                     for s in seeds]))
    frame = pd.concat(records,ignore_index=True)
    frame.to_csv(folder/'spread_evaluation.csv',index=False)
    metadata['test_evaluation_performed'] = True
    metadata['files_sha256']['spread_evaluation.csv'] = sha256(folder/'spread_evaluation.csv')
    (folder/'mappo_manifest.json').write_text(json.dumps(metadata,indent=2))
    print('Final held-out evaluation:',frame.groupby('controller')[['team_return','final_reward','landmarks_within_02']].mean().to_string(),flush=True)
    return metadata
