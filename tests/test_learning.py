import numpy as np
import torch
from rl_lab.checks import check_foundations,check_dqn,check_tit_for_tat,check_expected_payoffs
from rl_lab.social import read_payoffs,social_match


def test_foundation_edge_cases(definitions):
    check_foundations(*(definitions[name] for name in ['epsilon_greedy','td_target','q_update']))


def test_dqn_shapes_and_gradients(definitions):
    check_dqn(definitions['dqn_targets'],definitions['dqn_loss'])


def test_truncation_keeps_bootstrap(definitions):
    # A time-limit transition ends a rollout, but terminated=False is stored for learning.
    fn=definitions['dqn_targets']
    assert torch.allclose(fn(torch.tensor([2.]),torch.tensor([0.]),torch.tensor([10.]),.9),torch.tensor([11.]))
    assert torch.allclose(fn(torch.tensor([2.]),torch.tensor([1.]),torch.tensor([10.]),.9),torch.tensor([2.]))


def test_optimizer_changes_online_but_not_target(definitions):
    agent=definitions['DQNAgent'](seed=0)
    rng=np.random.default_rng(0)
    for i in range(70):
        agent.buffer.add(rng.normal(size=25),i%5,1.,rng.normal(size=25),i%9==0)
    online={k:v.clone() for k,v in agent.online.state_dict().items()}
    target={k:v.clone() for k,v in agent.target.state_dict().items()}
    assert np.isfinite(agent.learn())
    assert any(not torch.equal(v,online[k]) for k,v in agent.online.state_dict().items())
    assert all(torch.equal(v,target[k]) for k,v in agent.target.state_dict().items())
    agent.sync_target()
    assert all(torch.equal(v,agent.online.state_dict()[k]) for k,v in agent.target.state_dict().items())


def test_social_actions_payoffs_and_round_count(definitions):
    fn=definitions['tit_for_tat'];check_tit_for_tat(fn)
    frame=social_match('prisoners_dilemma',fn,'always_defect',nrounds=3)
    assert list(frame.a)==[0,1,1]
    assert list(frame.b)==[1,1,1]
    assert list(frame.reward_a)==[0,1,1]
    assert list(frame.reward_b)==[5,1,1]
    assert len(frame)==3


def test_stag_threshold_and_player_perspective(definitions):
    matrix=read_payoffs('stag_hunt');fn=definitions['expected_payoffs']
    check_expected_payoffs(fn,matrix)
    assert np.isclose(fn(matrix,2/3)[0],fn(matrix,2/3)[1])
    assert np.argmax(fn(matrix,.5))==1
    assert np.argmax(fn(matrix,.9))==0
