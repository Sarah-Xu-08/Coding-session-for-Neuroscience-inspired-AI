import copy
import json

import numpy as np
import pandas as pd
import pytest

from rl_lab.highway import highway_evaluate, train_highway
from rl_lab.highway_study import (training_rolling_mean, evaluation_rolling_mean,
                                  training_trace_summary, run_highway_study, load_cached_study)


def test_training_window_uses_only_latest_completed_episodes_in_each_run():
    frame=pd.DataFrame(dict(method=['DQN']*6,train_seed=[0,0,0,0,1,1],
        step=[30,10,25,20,8,9],return_=[2.,10.,100.,0.,4.,0.],
        complete=[True,True,False,True,True,True]))
    smoothed=training_rolling_mean(frame,window=2)
    assert list(smoothed.step)==[10,20,30,8,9]
    np.testing.assert_allclose(smoothed.rolling_mean,[10.,5.,1.,4.,2.])
    with pytest.raises(ValueError, match='positive integer'):
        training_rolling_mean(frame,window=0)


def test_evaluation_window_keeps_methods_and_seeds_separate():
    frame=pd.DataFrame(dict(method=['DQN','DQN','medium','DQN','medium'],
        seed=[0,0,0,1,0],step=[1000,0,1000,1000,0],return_=[20.,10.,2.,40.,4.]))
    smoothed=evaluation_rolling_mean(frame,window=2)
    np.testing.assert_allclose(smoothed.rolling_mean,[10.,15.,40.,4.,3.])


def test_run_alignment_carries_only_the_last_observed_estimate():
    traces=pd.DataFrame(dict(train_seed=[0,0,1,1],step=[10,30,20,40],rolling_mean=[2.,6.,10.,12.]))
    summary=training_trace_summary(traces)
    assert list(summary.index)==[20,30,40]
    np.testing.assert_allclose(summary['mean'],[6.,8.,9.])


def test_episode_log_accounts_for_every_training_step(definitions):
    agent,curve=train_highway('medium',definitions['rules'],steps=37,seed=17,
        eval_every=19,eval_seeds=[82000],verbose=False)
    episodes=agent.training_episodes
    assert episodes.length.sum()==37
    assert episodes.step.iloc[-1]==37
    assert list(curve.step)==[0,19,37]
    assert (agent.development_episodes.groupby('step').size()==1).all()
    assert episodes.complete.iloc[:-1].all()
    assert np.isfinite(curve.return_).all()
    assert agent.training_cpu_seconds>0 and agent.training_seconds>0
    # Evaluation restores the training RNG and never inserts table entries.
    state=copy.deepcopy(agent.rng.bit_generator.state);size=len(agent.q)
    first=highway_evaluate(agent,[93000,93001])
    second=highway_evaluate(agent,[93000,93001])
    pd.testing.assert_frame_equal(first,second)
    assert agent.rng.bit_generator.state==state and len(agent.q)==size


def test_final_traffic_cannot_be_used_as_development_traffic(definitions):
    with pytest.raises(ValueError,match='disjoint'):
        run_highway_study(definitions['rules'],definitions['DQNAgent'],
                          eval_seeds=[1,2],test_seeds=[2,3])


def test_incomplete_cache_cannot_be_presented_as_finished(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'complete':False}))
    with pytest.raises(RuntimeError,match='not finished'):load_cached_study(tmp_path)


def test_resume_reuses_only_matching_verified_runs(tmp_path):
    from run_highway_study import completed_run, digest
    spec={'steps':50,'seed':0}
    payload=tmp_path/'agent.npz';payload.write_bytes(b'example')
    (tmp_path/'completed.json').write_text(json.dumps(dict(specification=spec,
        files_sha256={'agent.npz':digest(payload)})))
    assert completed_run(tmp_path,spec)
    assert not completed_run(tmp_path,dict(spec,steps=100))
    payload.write_bytes(b'changed')
    assert not completed_run(tmp_path,spec)
