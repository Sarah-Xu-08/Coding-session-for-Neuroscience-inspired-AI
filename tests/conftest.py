import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
import rl_lab
import matplotlib
matplotlib.use('Agg')
from instructor_definitions import load


@pytest.fixture(scope='session')
def definitions():
    return load()
