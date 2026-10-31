"""Provided simulation and plotting support. Exercise answers live in the instructor notebook."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
os.environ.setdefault('MPLCONFIGDIR', str(RESULTS / '.matplotlib'))
sys.path.insert(0, str(ROOT / 'vendor'))
