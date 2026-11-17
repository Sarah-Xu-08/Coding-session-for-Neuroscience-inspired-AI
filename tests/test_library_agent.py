"""Exercise the actual notebook API example against the lab's HighwayEnv wrapper."""
import nbformat
import pytest
from pathlib import Path

from rl_lab.highway import ACTION_NAMES, make_highway
from rl_lab.library_dqn import SB3_DQN_KWARGS

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('train',[False,True])
def test_library_agent_can_predict_and_optionally_train(train):
    notebook=nbformat.read(ROOT/'notebooks/chapter3_student.ipynb',as_version=4)
    source=next(c.source for c in notebook.cells if c.id=='sb3-example')
    namespace={'make_highway':make_highway,'ACTION_NAMES':ACTION_NAMES,
               'SB3_DQN_KWARGS':SB3_DQN_KWARGS,'DEVICE':'cpu',
               'RUN_SB3_TRAINING':train,'budget':{'highway':288}}
    exec(source,namespace)
    agent=namespace['library_agent']
    assert 0<=int(namespace['action'])<5
    assert agent.num_timesteps==(288 if train else 0)
    assert agent._n_updates>0 if train else agent._n_updates==0
