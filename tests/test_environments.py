import copy
import hashlib
import json
import numpy as np
import pytest
from rl_lab import ROOT
from rl_lab.cafe import DessertCafe,cafe_episode,train_cafe
from rl_lab.highway import make_highway,PlaybackDQN,train_highway
from rl_lab.social import train_social,social_match


def test_cafe_inventory_accounting_and_seeded_customers():
    env=DessertCafe()
    env.reset(seed=12)
    traffic=copy.deepcopy(env.traffic)
    env.reset(seed=12)
    assert all(np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1]) for a,b in zip(traffic,env.traffic))
    _,rows=cafe_episode(env,lambda state,mask:int(np.flatnonzero(mask)[-1]),42)
    produced=np.array([2,2])+np.sum([r['produced'] for r in rows],axis=0)
    sold=np.sum([r['sales'] for r in rows],axis=0)
    waste=np.sum([r['waste'] for r in rows],axis=0)
    assert np.array_equal(produced,sold+waste)
    assert env.observation_space.contains(env.observation())
    assert rows[-1]['action']==0
    assert all(np.isclose(r['reward'],r['cash_profit']-.5*r['missed'].sum()) for r in rows)
    env.close()


def test_cafe_rejects_illegal_actions():
    env=DessertCafe();env.reset(seed=1);env.ingredients=0
    with pytest.raises(ValueError,match='infeasible'):env.step(1)
    env.close()


def test_cafe_and_social_training_use_injected_rules(definitions):
    cafe,history=train_cafe(definitions['rules'],episodes=20,seed=0)
    assert len(history)==20 and cafe.visits.sum()==160
    a,b,history=train_social('prisoners_dilemma',definitions['rules'],episodes=3,nrounds=4)
    assert len(history)==3 and len(social_match('prisoners_dilemma',a,b,nrounds=4))==4


def test_highway_features_and_rgb_rendering():
    env=make_highway(render=True)
    state,_=env.reset(seed=0)
    assert state.shape==(25,) and env.observation_space.contains(state)
    assert state.reshape(5,5)[0,1]==0.
    assert env.render().shape[-1]==3
    env.step(PlaybackDQN().act(state))
    env.close()


def test_sparse_highway_training_and_final_evaluation(definitions):
    agent,curve=train_highway('coarse',definitions['rules'],steps=16,eval_every=16,
                              eval_seeds=[81000],verbose=False)
    assert agent.training_steps==16 and len(agent.q)>0
    assert list(curve.step)==[0,16]


def test_vendored_source_is_unchanged():
    manifest=json.loads((ROOT/'vendor/PROVENANCE.json').read_text())
    for path,expected in manifest['files_sha256'].items():
        assert hashlib.sha256((ROOT/'vendor'/path).read_bytes()).hexdigest()==expected


def test_cafe_buttons_progress_close_and_reset():
    from rl_lab.practice import cafe_controls
    controls=cafe_controls(seed=2026)
    assert controls['env'].t==0
    for _ in range(8):controls['advance'].click()
    assert controls['env'].t==8 and controls['advance'].disabled
    controls['reset'].click()
    assert controls['env'].t==0 and not controls['advance'].disabled
    controls['env'].close()


def test_social_controls_callbacks(definitions,caplog):
    import logging
    import matplotlib.pyplot as plt
    from rl_lab.practice import dilemma_controls,spread_controls
    with caplog.at_level(logging.WARNING,logger='traitlets'):
        controls=dilemma_controls(definitions['tit_for_tat'])
        controls['a'].value=definitions['tit_for_tat']
        controls['b'].value='always_defect'
        controls['run'].click()
        spread=spread_controls(definitions['team_reward'])
        spread['slider'].value=20
        spread['strategy'].value='assignment'
        spread['slider'].value=spread['slider'].max
    assert not [record for record in caplog.records if record.name=='traitlets' and record.levelno>=logging.WARNING]
    plt.close('all')


@pytest.mark.parametrize('seed', [0, 7, 2026])
def test_spread_rewards_match_coverage_even_during_collisions(definitions, seed):
    from rl_lab.spread import make_spread, reset_spread, spread_positions
    env=make_spread(max_cycles=5, render=True)
    try:
        observations,_=reset_spread(env,seed=seed,layout='random')
        assert all(obs.shape==(18,) for obs in observations.values())
        # Intentionally overlap agent bodies; there must still be no local penalty.
        for i,agent in enumerate(env.unwrapped.world.agents):
            agent.state.p_pos=np.array([i*.04, i*.03])
        for step in range(5):
            names=list(env.agents)
            _,rewards,terminated,truncated,_=env.step({name:step%5 for name in names})
            positions,landmarks=spread_positions(env)
            expected=definitions['team_reward'](positions,landmarks)
            assert len(rewards)==3
            assert all(np.isclose(rewards[name],expected) for name in names)
        assert all(truncated.values()) and not any(terminated.values())
        assert env.render().shape[-1]==3
    finally:
        env.close()


def test_spread_controllers_start_from_the_same_world():
    from rl_lab.spread import spread_rollout
    nearest=spread_rollout('nearest')
    assigned=spread_rollout('assignment')
    np.testing.assert_equal(nearest['positions'][0],assigned['positions'][0])
    np.testing.assert_equal(nearest['landmarks'],assigned['landmarks'])
    assert len(nearest['history'])==len(assigned['history'])==60
    assert assigned['history'].team_reward.sum()>nearest['history'].team_reward.sum()
    for rollout in [nearest,assigned]:
        for name in ['agent_0','agent_1','agent_2']:
            np.testing.assert_allclose(rollout['history'][f'reward_{name}'],rollout['history'].team_reward)
