"""SB3 DQN for experiments; the notebook's hand-coded DQN remains a teaching exercise."""
import copy
from importlib.metadata import version
import random
import time

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from . import ROOT
from .highway import HIGHWAY_CONFIG, SparseTrafficQ, highway_evaluate, make_highway, train_highway
from .highway_study import (HighwayStudy, METHODS, baseline_evaluation, load_cached_study)

SB3_STUDY_DIR = ROOT / 'assets/reference_results/highway_sb3_study'
SB3_IMPLEMENTATION = 'stable_baselines3.DQN'
# Fixed before collecting the SB3 study. These are not the result of an SB3 pilot.
SB3_DQN_KWARGS = dict(
    learning_rate=1e-3, buffer_size=20_000, learning_starts=256, batch_size=64,
    gamma=.95, train_freq=4, gradient_steps=1, target_update_interval=400,
    exploration_fraction=.7, exploration_initial_eps=1., exploration_final_eps=.05,
    max_grad_norm=10., policy_kwargs={'net_arch': [64, 64]},
)


class LibraryDQNPolicy:
    """Use the lab's evaluation/playback interface with an actual SB3 model."""
    def __init__(self, model):
        self.model = model

    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

    def save(self, path):
        self.model.save(path)


class TrainingTrafficSeeds(gym.Wrapper):
    """Give SB3 auto-resets the same per-episode seed stream used by table training."""
    def __init__(self, env, seed):
        super().__init__(env)
        self.traffic_rng = np.random.default_rng(seed + 50_000)
        self.traffic_seeds = []

    def reset(self, *, seed=None, options=None):
        # SB3 seeds its policy/action space separately. Traffic uses this declared stream.
        traffic_seed = int(self.traffic_rng.integers(2**31))
        self.traffic_seeds.append(traffic_seed)
        return self.env.reset(seed=traffic_seed, options=options)


class StudyRecorder(BaseCallback):
    """Record actual episodes and evaluate a separate environment without changing training RNGs."""
    def __init__(self, steps, seed, eval_every, eval_seeds, verbose=False, on_checkpoint=None):
        super().__init__()
        self.steps, self.seed, self.eval_every = steps, seed, eval_every
        self.eval_seeds, self.show_progress = tuple(eval_seeds), verbose
        self.on_checkpoint = on_checkpoint
        self.episodes, self.development, self.curves = [], [], []
        self.episode_return, self.episode_length = 0., 0
        self.first_epsilon, self.last_info = 1., {}
        self.paused = self.cpu_paused = 0.

    def _on_training_start(self):
        self.started, self.cpu_started = time.perf_counter(), time.process_time()
        self.record(0)

    def training_times(self):
        return (time.perf_counter() - self.started - self.paused,
                time.process_time() - self.cpu_started - self.cpu_paused)

    def record(self, step):
        elapsed, cpu = self.training_times()
        evaluation_started, evaluation_cpu = time.perf_counter(), time.process_time()
        np_state, torch_state, random_state = np.random.get_state(), torch.get_rng_state(), random.getstate()
        training_mode = self.model.policy.training
        try:
            frame = highway_evaluate(LibraryDQNPolicy(self.model), self.eval_seeds)
        finally:
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
            random.setstate(random_state)
            self.model.policy.set_training_mode(training_mode)
        self.development.extend(frame.assign(step=step, train_seed=self.seed, method='DQN').to_dict('records'))
        self.curves.append(dict(step=step, return_=frame.return_.mean(), collision=frame.collision.mean(),
            train_seconds=elapsed, train_cpu_seconds=cpu, eval_episodes=len(frame),
            mean_duration_s=frame.duration_s.mean(), seen_fraction=np.nan))
        if self.on_checkpoint:
            self.on_checkpoint(step, self)
        self.paused += time.perf_counter() - evaluation_started
        self.cpu_paused += time.process_time() - evaluation_cpu
        if self.show_progress and (step % 10_000 == 0 or step == self.steps):
            print(f'SB3 DQN seed={self.seed} step={step:,} development reward={frame.return_.mean():.2f}', flush=True)

    def _episode(self, complete, epsilon):
        self.episodes.append(dict(episode=len(self.episodes) + 1, step=self.num_timesteps,
            return_=self.episode_return, length=self.episode_length,
            collision=int(self.last_info.get('crashed', False)), complete=complete,
            epsilon_start=self.first_epsilon, epsilon_end=epsilon, seen_fraction=np.nan,
            train_seed=self.seed, method='DQN'))
        self.episode_return, self.episode_length = 0., 0

    def _on_step(self):
        epsilon = 1. if self.num_timesteps <= self.model.learning_starts else self.model.exploration_rate
        if self.episode_length == 0:
            self.first_epsilon = epsilon
        self.episode_return += float(self.locals['rewards'][0])
        self.episode_length += 1
        self.last_info = self.locals['infos'][0]
        if self.locals['dones'][0]:
            self._episode(True, epsilon)
        # SB3 invokes step callbacks before that rollout's optimizer update.
        # Final evaluation is deferred until learn() has completed its final update.
        if self.num_timesteps % self.eval_every == 0 and self.num_timesteps < self.steps:
            self.record(self.num_timesteps)
        return True

    def _on_training_end(self):
        if self.episode_length:
            self._episode(False, self.model.exploration_rate)
        self.record(self.num_timesteps)


def train_library_dqn(steps=50_000, seed=0, eval_every=1000, eval_seeds=range(82000, 82020),
                      device='cpu', dqn_kwargs=None, verbose=False, on_checkpoint=None):
    """Train SB3, preserving true termination/truncation handling in its replay buffer."""
    config = copy.deepcopy(SB3_DQN_KWARGS)
    config.update(copy.deepcopy(dqn_kwargs or {}))
    if steps < 1 or eval_every < 1 or not len(eval_seeds):
        raise ValueError('Use positive budgets and at least one development scenario.')
    if steps % config['train_freq']:
        raise ValueError('Use a budget divisible by train_freq so SB3 collects the exact declared steps.')
    env = TrainingTrafficSeeds(make_highway(), seed)
    try:
        model = DQN('MlpPolicy', env, seed=seed, device=device, verbose=0, **config)
        recorder = StudyRecorder(steps, seed, eval_every, eval_seeds, verbose, on_checkpoint)
        model.learn(total_timesteps=steps, callback=recorder)
        agent = LibraryDQNPolicy(model)
        agent.training_steps = model.num_timesteps
        agent.training_seconds, agent.training_cpu_seconds = recorder.training_times()
        agent.training_episodes = pd.DataFrame(recorder.episodes)
        agent.development_episodes = pd.DataFrame(recorder.development)
        agent.training_traffic_seeds = env.traffic_seeds.copy()
        agent.training_config = config
        return agent, pd.DataFrame(recorder.curves)
    finally:
        env.close()


def sb3_study_metadata(steps, seeds, eval_every, eval_seeds, test_seeds, dqn_kwargs, table_kwargs):
    return dict(experiment_id='highway_sb3_study_v1', complete=True,
        steps_per_method=steps, training_seeds=list(seeds), development_seeds=list(eval_seeds),
        test_seeds=list(test_seeds), eval_every=eval_every, methods=list(METHODS),
        dqn_implementation=SB3_IMPLEMENTATION, dqn_kwargs=copy.deepcopy(dqn_kwargs),
        table_kwargs=copy.deepcopy(table_kwargs), config=copy.deepcopy(HIGHWAY_CONFIG),
        versions={name: version(name) for name in ['stable-baselines3','torch','numpy','gymnasium','highway-env']},
        source='SB3 DQN and tabular policies trained in this notebook',
        sb3_settings_selection='Fixed in advance; no SB3 hyperparameter pilot.',
        checkpoint_timing='Development checkpoints precede the associated optimizer update; final evaluation follows the final optimizer update.',
        model_selection='Final fixed-budget policy, not the best test checkpoint.')


def run_sb3_highway_study(rules, steps=50_000, seeds=range(5), eval_every=1000,
                          eval_seeds=range(82000,82020), test_seeds=range(93000,93100),
                          device='cpu', dqn_kwargs=None, table_kwargs=None):
    seeds, eval_seeds, test_seeds = list(seeds), list(eval_seeds), list(test_seeds)
    if not seeds or not eval_seeds or not test_seeds or set(eval_seeds) & set(test_seeds):
        raise ValueError('Use nonempty training seeds and disjoint development/test traffic.')
    config = copy.deepcopy(SB3_DQN_KWARGS)
    config.update(dqn_kwargs or {})
    table_kwargs = dict(table_kwargs or {'alpha': .3})
    agents, training, evaluation, curves, scores = {}, [], [], [], []
    for seed in seeds:
        for kind in METHODS:
            if kind == 'DQN':
                agent, curve = train_library_dqn(steps, seed, eval_every, eval_seeds,
                    device=device, dqn_kwargs=config)
            else:
                agent, curve = train_highway(kind, rules, steps=steps, seed=seed,
                    eval_every=eval_every, eval_seeds=eval_seeds, agent_kwargs=table_kwargs, verbose=False)
            agents[(kind, seed)] = agent
            training.append(agent.training_episodes)
            evaluation.append(agent.development_episodes)
            curves.append(curve.assign(method=kind, seed=seed))
            scores.append(highway_evaluate(agent, test_seeds).assign(method=kind, train_seed=seed,
                train_seconds=agent.training_seconds, train_cpu_seconds=agent.training_cpu_seconds,
                table_states=len(agent.q) if isinstance(agent, SparseTrafficQ) else np.nan))
    return HighwayStudy(*(pd.concat(part, ignore_index=True) for part in [training,evaluation,curves,scores]),
        baseline_evaluation(eval_seeds,test_seeds),
        sb3_study_metadata(steps,seeds,eval_every,eval_seeds,test_seeds,config,table_kwargs), agents)


def load_sb3_study(folder=SB3_STUDY_DIR):
    study = load_cached_study(folder)
    if study.metadata.get('dqn_implementation') != SB3_IMPLEMENTATION:
        raise ValueError('Expected an actual Stable-Baselines3 DQN study, not hand-coded DQN scores.')
    return study


def load_sb3_study_dqn(seed=0, folder=SB3_STUDY_DIR):
    return LibraryDQNPolicy(DQN.load(folder / 'runs' / f'DQN_seed{seed}' / 'agent.zip', device='cpu'))
