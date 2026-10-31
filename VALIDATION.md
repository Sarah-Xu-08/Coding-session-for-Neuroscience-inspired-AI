# Local validation · 2026-10-31

Validated on the recording machine: Apple Silicon (`arm64`), macOS, Python 3.11,
CPU PyTorch 2.5.1. Direct dependencies are in `environment.yml`; the installed pip
versions are recorded in `requirements-macos-arm64.lock.txt`.

| Check | Result |
|---|---|
| Environment and all four tasks | Passed, including HighwayEnv RGB rendering and checkpoint loading |
| Automated behavior checks | 16 passed |
| Fresh-kernel smoke execution | Instructor and temporarily completed student flows passed; includes new short highway training |
| Default teaching execution | Instructor and temporarily completed student flows passed in fresh kernels |
| Instructor presentation outputs | 7 inline PNG charts, 1 GIF animation, 3 widget views |
| Student teaching outputs during verification | 7 inline PNG charts and 1 GIF; widget display skipped only in the temporary headless check |
| Control callbacks | Café advance/close/reset, dilemma strategy selection/play, and Stag Hunt slider passed |
| Distributed student workbook | 27 code cells; no saved outputs, instructor-only cells, or solution metadata |
| Upstream games | All bundled source hashes match the supplied manifest |

The maintained instructor notebook includes the default teaching outputs. The
distributed student notebook remains unfinished by design. Verification injects
solutions into a temporary in-memory copy; it never fills the distributed file.
Live controls remain enabled in both distributed notebooks. Student widget display
is skipped in the headless check because of an nbclient widget-state issue; control
callbacks and instructor widget views are checked separately.

The bundled café reference was trained locally for 60,000 days, seed 0. The cached
highway teaching comparison was trained locally for 2,000 interactions per method,
seed 0, with separate development and held-out seeds. Full multi-seed experiments
are available but were not used to claim algorithm superiority.

The supplied 20,000-step DQN checkpoint was re-evaluated locally on seeds
92000–92019. Its mean return was 19.415 and collision fraction 0.10, matching the
original recorded evaluation. See `assets/reference_results/manifest.json` and
the per-episode CSV for the measured results. This checkpoint remains separate
from the short-budget comparison.

SciPy 1.14.1 loads and runs on this Mac; the initially tried 1.15.3 wheel did not.
The notebook explicitly enables inline Matplotlib plotting. The environment check,
tests, and verification commands are documented in the README. Google Colab and
Windows have not been validated in this release.
