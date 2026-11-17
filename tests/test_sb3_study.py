import copy
import pickle
import random

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import DQN

import rl_lab.library_dqn as library


class TinyTraffic(gym.Env):
    observation_space=gym.spaces.Box(-3.,3.,shape=(25,),dtype=np.float32)
    action_space=gym.spaces.Discrete(5)

    def __init__(self, terminal=False):
        self.terminal=terminal

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps=0
        return np.zeros(25,np.float32),{}

    def step(self, action):
        self.steps+=1
        done=self.steps==2
        return np.zeros(25,np.float32),1.,done and self.terminal,done and not self.terminal,{'crashed':done and self.terminal}


def fake_evaluation(agent,seeds):
    agent.act(np.zeros(25,np.float32))
    # Evaluation is allowed to use randomness without perturbing training streams.
    np.random.random();random.random();torch.rand(1)
    return pd.DataFrame([dict(seed=s,return_=2.,collision=0,mean_speed_m_s=20.,
        steps=2,duration_s=2.,seen_fraction=np.nan,lane_change_requests=0) for s in seeds])


@pytest.mark.parametrize('terminal',[False,True])
def test_sb3_records_exact_steps_and_bootstraps_only_truncation(monkeypatch,terminal,tmp_path):
    monkeypatch.setattr(library,'make_highway',lambda:TinyTraffic(terminal))
    monkeypatch.setattr(library,'highway_evaluate',fake_evaluation)
    agent,curve=library.train_library_dqn(steps=12,eval_every=4,eval_seeds=[82000],
        dqn_kwargs={'learning_starts':0,'batch_size':2,'buffer_size':32})
    assert isinstance(agent.model,DQN) and agent.model._n_updates==3
    assert agent.training_episodes.length.sum()==12
    assert agent.training_episodes.complete.all()
    assert list(curve.step)==[0,4,8,12]
    assert (agent.development_episodes.groupby('step').size()==1).all()
    # SB3 stores the time-limit flag and removes it from the sampled terminal mask.
    sample=agent.model.replay_buffer._get_samples(np.array([1]))
    assert sample.dones.item()==float(terminal)
    path=tmp_path/'agent.zip';agent.save(path)
    loaded=library.LibraryDQNPolicy(DQN.load(path,device='cpu'))
    assert loaded.act(np.zeros(25,np.float32))==agent.act(np.zeros(25,np.float32))


def test_checkpoint_evaluation_preserves_rng_weights_and_policy_mode(monkeypatch):
    monkeypatch.setattr(library,'highway_evaluate',fake_evaluation)
    env=TinyTraffic();model=DQN('MlpPolicy',env,buffer_size=32,seed=0,device='cpu')
    recorder=library.StudyRecorder(12,0,4,[82000]);recorder.init_callback(model)
    model.policy.set_training_mode(True)
    np_before=pickle.dumps(np.random.get_state());random_before=random.getstate()
    torch_before=torch.get_rng_state().clone()
    weights=copy.deepcopy(model.policy.state_dict())
    recorder._on_training_start()
    assert pickle.dumps(np.random.get_state())==np_before and random.getstate()==random_before
    assert torch.equal(torch.get_rng_state(),torch_before) and model.policy.training
    assert all(torch.equal(value,weights[key]) for key,value in model.policy.state_dict().items())
    env.close()


def test_experiment_budget_cannot_silently_round_up():
    with pytest.raises(ValueError,match='divisible'):
        library.train_library_dqn(steps=13,eval_every=4,eval_seeds=[82000])


def test_sb3_cache_loader_rejects_hand_coded_results(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(library,'load_cached_study',lambda folder:SimpleNamespace(metadata={}))
    with pytest.raises(ValueError,match='actual Stable-Baselines3'):
        library.load_sb3_study()
