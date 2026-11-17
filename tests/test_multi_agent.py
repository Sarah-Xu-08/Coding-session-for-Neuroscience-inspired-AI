import copy

import numpy as np
import pandas as pd
import pytest
import torch
from skrl.multi_agents.torch.mappo import MAPPO

from rl_lab.mappo import (GROUP, build_mappo, load_mappo, train_mappo,
                          pack_local_observations, finalize_mappo, local_distance_features,
                          evaluate_mappo, REFERENCE)
from rl_lab.social import social_match
from rl_lab.social_reference import load_social_reference
from rl_lab.spread import spread_rollout


def test_library_mappo_really_trains_and_reloads(tmp_path, monkeypatch):
    import rl_lab.mappo as mappo
    # Exercise the final-evaluation path without inspecting the lesson's reserved tests.
    monkeypatch.setattr(mappo, 'DEV_SEEDS', [77000001,77000002])
    monkeypatch.setattr(mappo, 'TEST_SEEDS', [88000001,88000002])
    torch.set_num_threads(1)
    initial = build_mappo(seed=0)
    before = copy.deepcopy(initial.policies[GROUP].state_dict())
    first_stage = train_mappo(tmp_path, joint_steps=120, num_envs=2, seed=0)
    assert not first_stage['test_evaluation_performed']
    metadata = train_mappo(tmp_path, joint_steps=240, num_envs=2, seed=0, resume=True, evaluate_test=True)
    policy, loaded_metadata = load_mappo(tmp_path)
    assert isinstance(policy.agent, MAPPO)
    assert metadata == loaded_metadata
    assert metadata['joint_steps'] == 240
    assert metadata['individual_actions'] == 720
    assert len(metadata['training_stages']) == 2
    assert any(not torch.equal(v, policy.agent.policies[GROUP].state_dict()[k])
               for k, v in before.items())
    assert policy.agent.values[GROUP].num_observations == 54
    assert policy.agent.policies[GROUP].num_observations == 18
    assert metadata['parameter_sharing']
    with pytest.raises(ValueError, match='final test is already recorded'):
        finalize_mappo(tmp_path)
    with pytest.raises(ValueError, match='Final test has already been inspected'):
        train_mappo(tmp_path,joint_steps=360,num_envs=2,resume=True)
    # Reproduce the first recorded final test trajectory from the reloaded model.
    recorded = pd.read_csv(tmp_path/'spread_evaluation.csv').iloc[0]
    rollout = spread_rollout('mappo', seed=int(recorded.seed), layout='random', policy=policy)
    assert np.isclose(rollout['history'].team_reward.sum(), recorded.team_return)
    assert np.isclose(rollout['rewards'][-1], recorded.final_reward)
    assert not set(metadata['test_seeds']) & set(metadata['development_seeds'])
    (tmp_path/'mappo_training.csv').write_text('changed')
    with pytest.raises(ValueError, match='reference file changed'):
        load_mappo(tmp_path)


def test_supplied_mappo_uses_local_actors_without_critic_at_execution(monkeypatch):
    policy, metadata = load_mappo()
    def no_critic(*args, **kwargs):
        raise AssertionError('Evaluation must not call the global-state critic')
    for value in policy.agent.values.values():
        monkeypatch.setattr(value, 'compute', no_critic)
    key = next(iter(policy.agent.policies))
    before = copy.deepcopy(policy.agent.policies[key].state_dict())
    learned = spread_rollout('mappo', policy=policy)
    nearest = spread_rollout('nearest')
    np.testing.assert_array_equal(learned['positions'][0], nearest['positions'][0])
    np.testing.assert_array_equal(learned['landmarks'], nearest['landmarks'])
    assert len(learned['history']) == 60
    for i in range(3):
        np.testing.assert_allclose(learned['history'][f'reward_agent_{i}'], learned['history'].team_reward)
    assert all(torch.equal(v, policy.agent.policies[key].state_dict()[k])
               for k, v in before.items())
    assert metadata['library'] == 'skrl'


def test_shared_policy_batch_preserves_agent_views_and_has_one_optimizer():
    observations = [{name: np.full(18, i*10+j) for j,name in enumerate(['agent_0','agent_1','agent_2'])}
                    for i in range(2)]
    packed = pack_local_observations(observations)
    np.testing.assert_array_equal(packed[:,0], [0,1,2,10,11,12])
    assert packed.shape == (6,18)
    agent = build_mappo(num_envs=2,training=True)
    assert list(agent.policies) == list(agent.values) == list(agent.optimizers) == [GROUP]
    assert agent.memories[GROUP].num_envs == 6


def test_timeout_bootstraps_scaled_training_reward_but_preserves_reported_reward(monkeypatch):
    agent = build_mappo(num_envs=1,training=True)
    local = {GROUP:torch.zeros(3,18)}
    shared = torch.zeros(3,54)
    actions,_,_ = agent.act(local,timestep=0,timesteps=60)
    monkeypatch.setattr(agent.values[GROUP], 'act',
                        lambda *args,**kwargs:(torch.full((3,1),3.),None,{}))
    environment_rewards = torch.full((3,1),-2.)
    agent.record_transition(local,actions,{GROUP:environment_rewards},local,
                            {GROUP:torch.zeros(3,1,dtype=torch.bool)},
                            {GROUP:torch.ones(3,1,dtype=torch.bool)},
                            {'shared_states':shared,'shared_next_states':shared},
                            timestep=0,timesteps=60)
    np.testing.assert_allclose(agent.memories[GROUP].get_tensor_by_name('rewards')[0], .1*(-2.)+.99*3.)
    np.testing.assert_array_equal(environment_rewards, -2.)


def test_distance_features_use_only_information_in_each_local_observation():
    from rl_lab.spread import make_spread,reset_spread,spread_positions
    env = make_spread()
    try:
        observations,_ = reset_spread(env,seed=17,layout='random')
        names = ['agent_0','agent_1','agent_2']
        local = torch.tensor(np.stack([observations[n] for n in names]))
        features = local_distance_features(local).numpy()
        agents,landmarks = spread_positions(env)
        distances = np.linalg.norm(agents[:,None,:]-landmarks[None,:,:],axis=-1)
        np.testing.assert_allclose(features[:,18:21],distances,rtol=1e-6,atol=1e-6)
        for i in range(3):
            np.testing.assert_allclose(features[i,21:24],np.delete(distances,i,axis=0).min(axis=0),rtol=1e-6,atol=1e-6)
        np.testing.assert_array_equal(env.state().reshape(3,18),local)
    finally:
        env.close()


def test_reference_mappo_reproduces_heldout_scores_and_covers_landmarks():
    """Guard lesson quality as well as successful code execution."""
    torch.set_num_threads(1)
    policy,metadata = load_mappo()
    assert metadata['format_version'] == 4 and metadata['test_evaluation_performed']
    recorded = pd.read_csv(REFERENCE/'spread_evaluation.csv')
    expected = recorded[recorded.controller=='MAPPO (skrl)'].reset_index(drop=True)
    actual = evaluate_mappo(policy,metadata['test_seeds'])
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,atol=1e-6,rtol=1e-6)
    nearest = recorded[recorded.controller=='nearest']
    assert actual.team_return.mean() > nearest.team_return.mean()
    assert actual.landmarks_within_02.mean() >= 2.5
    demo = spread_rollout('mappo',policy=policy)
    distances = np.linalg.norm(demo['positions'][-1,:,None,:]-demo['landmarks'][None,:,:],axis=-1)
    assert (distances.min(axis=0)<=.2).sum() == 3
    assert np.linalg.norm(demo['positions'],axis=-1).max() < 3


def test_frozen_independent_q_learning_uses_two_distinct_tables(definitions):
    a, b, history, metadata = load_social_reference(definitions['rules'])
    assert a is not b and a.q is not b.q and len(a.q) > 0 and len(b.q) > 0
    assert len(history) == metadata['episodes'] == 3000
    before = {state: values.copy() for state, values in a.q.items()}
    match = social_match('prisoners_dilemma', a, b, nrounds=20, epsilon=0., learn=False)
    assert len(match) == 20 and len(before) == len(a.q)
    for state, values in before.items():
        np.testing.assert_array_equal(values, a.q[state])
    assert metadata['joint_decisions'] == 60000
    from rl_lab.social_reference import REFERENCE
    pd.testing.assert_frame_equal(match, pd.read_csv(REFERENCE/'independent_q_selfplay.csv'), check_dtype=False)


def test_learned_policy_controls_preserve_frozen_evaluation(definitions, caplog):
    import logging
    import matplotlib.pyplot as plt
    from rl_lab.practice import dilemma_controls, spread_controls
    a, b, *_ = load_social_reference(definitions['rules'])
    policy, _ = load_mappo()
    learned = spread_rollout('mappo', policy=policy)
    with caplog.at_level(logging.WARNING, logger='traitlets'):
        controls = dilemma_controls(definitions['tit_for_tat'], learned=(a,b))
        controls['a'].value = a
        controls['b'].value = b
        assert controls['run'].description == 'Play 20 rounds'
        controls['run'].click()
        replay = spread_controls(definitions['team_reward'], learned=learned)
        replay['strategy'].value = 'mappo'
        replay['slider'].value = 60
        replay['strategy'].value = 'assignment'
    assert not [r for r in caplog.records if r.name == 'traitlets' and r.levelno >= logging.WARNING]
    plt.close('all')
