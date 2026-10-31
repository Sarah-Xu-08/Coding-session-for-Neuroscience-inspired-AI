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
    print('Python:', sys.executable)
    print('Platform:', platform.system(), platform.machine())
    for name in ['numpy','torch','gymnasium','highway-env','pettingzoo','jupyterlab','ipywidgets']:
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
    for name in ['prisoners_dilemma','stag_hunt']:
        assert read_payoffs(name).shape == (2,2,2)
    assert (ROOT / 'assets/figures/four_tasks_overview.png').is_file()
    print('Environment checks passed: four environments, RGB rendering and checkpoint loading.')


if __name__ == '__main__':
    check()
