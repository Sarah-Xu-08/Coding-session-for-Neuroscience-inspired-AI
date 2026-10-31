"""Check this kernel/interpreter, dependencies, rendering and bundled assets."""
import platform
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check():
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f'This lab is tested on Python 3.11; active interpreter is {sys.executable}')
    import rl_lab
    import numpy as np
    import torch
    import ipywidgets
    from rl_lab.cafe import DessertCafe
    from rl_lab.highway import make_highway, PlaybackDQN
    from rl_lab.social import read_payoffs
    from rl_lab.spread import make_spread
    print('Python:', sys.executable)
    print('Platform:', platform.system(), platform.machine())
    for name in ['numpy','torch','gymnasium','highway-env','pettingzoo','mpe2','jupyterlab','ipywidgets']:
        print(f'{name}: {version(name)}')
    env = DessertCafe()
    observation, _ = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    env.close()
    env = make_highway(render=True)
    state, _ = env.reset(seed=0)
    assert env.render().ndim == 3
    action = PlaybackDQN().act(state)
    assert 0 <= action < 5
    env.step(action)
    env.close()
    assert read_payoffs('prisoners_dilemma').shape == (2,2,2)
    env = make_spread(render=True)
    observations, _ = env.reset(seed=0)
    assert all(obs.shape == (18,) for obs in observations.values())
    _, rewards, _, _, _ = env.step({name: 0 for name in env.agents})
    assert len(set(rewards.values())) == 1
    assert env.render().shape[-1] == 3
    env.close()
    for name in ['cafe','highway','prisoners_dilemma','simple_spread']:
        assert (ROOT / f'assets/figures/task_{name}.png').is_file()
    print('Environment checks passed: four environments, RGB rendering and checkpoint loading.')


if __name__ == '__main__':
    check()
