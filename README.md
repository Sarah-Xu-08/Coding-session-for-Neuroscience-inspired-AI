# Neuroscience-inspired AI · Chapter 3 coding lab

A local NumPy + PyTorch introduction to reinforcement learning, designed for a
30-minute TA video, with additional student pause time, on an Apple Silicon Mac.
The student workbook begins with a roadmap connecting the activities to Chapter 3.
The recording covers a Dubai Chewy Cookie café, highway navigation and Simple
Spread. Prisoner's Dilemma is optional self-study in Appendix A.4, contrasting
conflicting individual incentives with a shared team objective. Longer experiments are optional.
Each game starts with a black-and-white line-art overview. Colour marks only the
teaching focus: the brown cookie and matching curve, the controlled car, game choices,
or coverage assignments and the shared team reward.

Open **[the student workbook](notebooks/chapter3_student.ipynb)** for practice.
Start with café decisions, introduce NumPy through its Q-table, then introduce
PyTorch through the highway Q-network and its training update. Immediately afterward,
use Stable-Baselines3 (SB3) DQN for the driving experiment.
Hand-code to understand or change a learning rule; use a maintained library for
standard agents and experimental baselines, trying policy settings and callbacks
before writing a new agent. The student workbook contains detailed explanations,
missing-line TODOs and checks; there is no solution switch. The instructor notebook,
recording script, preparation notes and PDF exports stay local and are Git-ignored.

## On the prepared Desktop Mac

The project folder is `~/Desktop/Coding-session-for-Neuroscience-inspired-AI`.
Its dedicated Python 3.11 environment is in `.conda/`; your base environment is separate.

Double-click **`start_lab.command`** to open the student workbook in JupyterLab.
On the recording machine, the launcher uses the local instructor notebook when present.
Alternatively, in Terminal:

```bash
cd ~/Desktop/Coding-session-for-Neuroscience-inspired-AI
conda activate ./.conda
python -m jupyterlab notebooks/chapter3_student.ipynb
```

Choose **Python (RL Chapter 3)** as the notebook kernel. The first cell prints the
active Python executable so you can verify it ends in this project's `.conda/bin/python`.
The repository is already connected to the private GitHub remote.

## Recreate the environment on another Mac

Install Conda for Apple Silicon, clone the private repository using your GitHub
credentials, then run these commands from the repository root:

```bash
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
python -m ipykernel install --user --name rl-chapter3 --display-name "Python (RL Chapter 3)"
python scripts/check_environment.py
python -m jupyterlab
```

The environment is the project-local `.conda/` directory. **Python (RL Chapter 3)**
is its Jupyter kernel label, rather than a named Conda environment. User-level
kernel registration makes it visible when Jupyter is launched from base Anaconda
as well. Refresh the kernel list after registering; an interpreter picker may
instead show the `.conda/bin/python` path.

The environment uses CPU PyTorch. No CUDA, MPS setup, API key, external dataset,
Google account, or native simulator window is required. Highway images are rendered
as RGB arrays inside the notebook, using an offscreen surface even with the SDL
dummy driver. The controlled car is brown; other traffic is grey. Installations
happen outside the notebook.
The pip section explicitly uses official PyPI because the package mirror available
during setup was missing a HighwayEnv file.

`environment.yml` pins the direct dependencies. `requirements-macos-arm64.lock.txt`
records all installed pip package versions from the tested Mac. SciPy 1.14.1 was
selected after the 1.15.3 wheel failed to load on the recording machine. The environment
pins MPE2 0.0.1, which shares the existing pygame 2.6.1 dependency with HighwayEnv.
The Simple Spread activity uses the actual simulator and its parallel API.
The environment has been validated on this Mac; Windows and Google Colab adaptation
are future work.
See [VALIDATION.md](VALIDATION.md) for the completed checks and their scope.

## How to use the student workbook

Run cells in order and follow the same Task 1/2/3 labels as the video. Replace
each marked `...` expression, run the function cell, then run its check cell. An
unfinished function intentionally raises `NotImplementedError` when called. Run All
is suitable after the three required tasks are complete; optional exercises are
skipped by default. Assertions explain the behavior to check; they do not
select or import instructor implementations.

| Activity | What students practice |
|---|---|
| Task 1 | Two expressions: masked tabular TD target and Q-value update; exploration is provided |
| Interpretation | Interpret café decisions, next-hour preparation, reward versus cash profit |
| Task 2 | One expression: vector DQN targets, masking true termination |
| Task 3 | Two expressions: select action values and calculate mean Huber loss |
| Library and experiment | Instantiate SB3 DQN, then compare measured SB3 training runs with tabular agents |
| Optional Task 4 | Complete a separate tit-for-tat practice function |
| Optional Task 5 | Complete a shared team coverage reward from pairwise distances |

The recording ends at **4 · Connect value learning and multi-agent decisions**.
Appendix activities are outside the video.
Simple Spread uses provided MAPPO and reward functions. The Prisoner's Dilemma
payoff plot, tit-for-tat rule and Independent Q-learning comparison are in Appendix
A.4; run that section after setup and Task 1 for self-study. Unfinished optional
exercises do not block the core flow. Enable
`RUN_OPTIONAL_PRACTICE` to attempt them. `RUN_EXTRA_ANALYSIS` enables the full café
policy table, ordinary-code café alternative and older reference-policy playback.

Stable-Baselines3 2.7.1 and skrl 1.4.3 are included in the environment. Existing
checkouts need these additions before running the revised notebooks (already
installed on the prepared recording Mac):

```bash
conda activate ./.conda
python -m pip install --index-url https://pypi.org/simple stable-baselines3==2.7.1 skrl==1.4.3
```

The library demonstration creates an explicitly **untrained** DQN by default.
Set `RUN_SB3_TRAINING=True` to call `.learn()` using the selected highway budget.
A short run demonstrates the API, not reliable driving. The library example uses
the same settings and HighwayEnv wrapper as the later SB3 experiment, which trains
fresh models or loads its separately recorded results. The API example's untrained
prediction is not an experimental result. The hand-coded DQN remains an inspectable
learning exercise. Its earlier study is preserved as historical material.

Appendix A.4's Prisoner's Dilemma reuses Q-learning with **two independently learning players**:
each has a separate Q-table and private payoff. That optional section loads a pair
trained for 3,000 twenty-round self-play episodes, then freezes learning and
exploration for partner comparisons. Tit-for-tat, always cooperate and always
defect remain clearly labelled fixed rules. Learning does not guarantee cooperation.

Simple Spread introduces **MAPPO (Multi-Agent Proximal Policy Optimization)** from
the existing **skrl 1.4.3** library. During training, each actor receives its own
18-value observation, while the centralized critic receives the 54-value global
state. The three interchangeable agents share actor/critic parameters. Each actor
call uses its own local observation and can choose a different action. Descriptive
distance features are derived from these existing observations; no assigned targets
or expert actions are provided. At execution only the actor is called. The library handles the algorithm;
`rl_lab/mappo.py` supplies networks, environment batching and checkpoint I/O, with
no locally implemented PPO loss or update. The provided reward function describes
the environment. A critic with team context does not identify each agent's causal
contribution, so the credit-assignment question remains part of this example.

The main cell loads a genuine library checkpoint, runs its actors in the simulator,
and compares it with fixed nearest-landmark steering from the same start. The
controls also offer a hand-coded centralized assignment reference. Both fixed
controllers use global positions; neither is presented as learned MAPPO.
The café buttons reveal outcomes after each decision. The Dilemma selectors
compare frozen policies, while the Spread selector/slider replays captured actual
simulator trajectories, including MAPPO.
Each has an ordinary-code alternative. Widgets need a live Jupyter kernel and do not
run in the static GitHub preview.

The café sells one product, **Dubai Chewy Cookie**, shown in brown. State is `(hour, stock)`;
`Q[hour, stock, action]` has shape `(11, 9, 3)`, including the terminal hour.
Actions prepare 0, 2 or 4 cookies for the next hour. Ingredients are always
available; only Poisson customer demand is random. There is no ingredient budget
or delivery failure. Capacity and final-hour action masks remain. Selling hours are
10:00–19:00, closing at 20:00. Expected demand has a larger peak at 12:00
(4 customers/hour) and a smaller one at 18:00 (3 customers/hour). Individual
random days can differ from this two-peak mean pattern.

The core results figure follows one real day: known mean versus sampled demand,
stock and sales, and learned versus model-optimal preparation at the SAME observed
hour and stock. The optional Appendix A.2 policy table shows hour × stock, without slicing any
hidden variables. Brown numbers show portions to prepare; `0` means wait. `?` means
no training update for that state, not forbidden actions. The reference can answer
hypothetical stock/hour questions even where a fixed opening stock prevents visits.
The reference knows the demand distribution, not future customer counts. It uses
exact Poisson probabilities and backward induction, provided in `rl_lab/cafe_reference.py`.
The forecast rule remains a separate heuristic. Evaluation uses matched customer days.
When enabled, full hour/stock/visit/action comparisons are saved to `cafe_policy_comparison.csv`.

## Training and recording

The default video run trains the café live and loads the **SB3 highway study**:
50,000 interactions per policy, four learned methods, five training seeds, plus two
non-learning baselines. All highway results are local measurements, with raw CSVs
and every final model included. Loading the cache avoids long training during the video.

Set `RUN_HIGHWAY_TRAINING=True` to train the tabular agents and actual SB3 DQN.
The hand-coded DQN's targets/loss remain the exercises; SB3 supplies its own update.
The setup
budgets below apply only to **new live training**; changing mode alone does not
turn training on or change the cached dataset.

| Mode | Café days | Live highway interactions / policy | Training seeds | Use |
|---|---:|---:|---|---|
| `smoke` | 200 | 288 | 0 | Code paths and SB3 optimizer updates, not learning quality |
| `teaching` | 20,000 | 2,000 | 0 | Short live demonstration, not a ranking |
| `experiment` | 60,000 | 50,000 | 0, 1, 2, 3, 4 | Full sequential comparison |

`RUN_SOCIAL_TRAINING` enables a separate fresh Independent Q-learning run in
Appendix A.6. Appendix A.4 loads the prepared learned pair without that flag. MAPPO training
stays outside the notebook and recording route. To reproduce the supplied policies
in a separate folder:

```bash
.conda/bin/python scripts/run_multi_agent.py --algorithm all --output results/my_marl --joint-steps 3000000
# Review the development curve and trajectories before final evaluation.
.conda/bin/python scripts/run_multi_agent.py --algorithm mappo --output results/my_marl --finalize-only
```

The first command does not evaluate the reserved final test. `--resume` continues
from `last.pt` to a larger total `--joint-steps` budget, restarting environments at
the stage boundary and preserving optimizer/RNG state. Once the final test is
recorded, the runner refuses to resume that experiment. Use `--algorithm iql` or
`--algorithm mappo` separately if needed. `--evaluate-test` is an explicit shortcut
for a preplanned final run, after development choices have been fixed.

The improved setup uses one shared actor and one shared centralized critic via
skrl's MAPPO class. Environment-major, agent-minor batching combines all agents'
experience under **one optimizer**, rather than aliasing models across independent
optimizers. Each actor takes 18 local values and appends six locally computable
distances (own and closest teammate distance to each landmark). The critic derives
corresponding features from the 54-value global state. Two 64-unit ReLU layers use
feature/hidden LayerNorm. The library supplies all policy and value updates.

The 60-step cutoff is a **truncation**: library time-limit bootstrapping retains
future value. Training scales rewards by 0.1 for numerical conditioning; all
scores and plots retain the original coverage-reward units. This positive constant
does not change the reward objective. The critic is unnormalized/unclipped; a wide
value-clipping window works around skrl 1.4.3's dictionary-valued flag check. The
actor's PPO ratio clip remains 0.2. Settings are recorded in the manifest.

Training uses 16 batched environments, seed 0, with 80% random starts and 20% the
practiced clustered teaching start. Training reset seeds lie in [1,000,000,000,
2,000,000,000). Checkpoint selection uses 16 development starts (171000–171015).
Final evaluation uses 50 reserved random starts (271000–271049), identical for all
controllers. The clustered scene, seed 2026, is a **practiced illustration**, not
held-out evidence. One training run does not measure variation across training seeds.

The chosen run completed **3,000,000 joint steps**; its best development checkpoint
was at **2,400,960 steps**. Three development configurations were trained for
3,000,000 steps each (9,000,000 total across the search); the final test was reserved
until configuration and checkpoint selection were fixed. Their settings and curves
are included in `development_search.json` and `development_2.csv` through
`development_4.csv`. The selected policy covers all three landmarks in the recording
scene. On the same 50 new random starts, measured mean team returns are:

| Controller | Mean return | Mean final landmarks covered (of 3) |
|---|---:|---:|
| Archived previous MAPPO | −86.22 | 0.64 |
| Improved MAPPO | −33.90 | 2.70 |
| Nearest steering (fixed) | −64.73 | 1.78 |
| Centralized assignment (fixed) | −21.29 | 3.00 |

Coverage means an agent finishes within 0.2 world units of a landmark. Improved
MAPPO covers all three in 41/50 starts. This diagnostic does not change the reward.
The old model was scored on these same starts only after selection was frozen;
its original evaluation remains archived. The fixed planner still outperforms
MAPPO; these results demonstrate improvement without claiming optimal coordination.

`assets/reference_results/multi_agent/` contains the selected checkpoint, training
records, final comparisons and version/hash manifests. The previous 960,000-step
model and its integration source are preserved under
`assets/reference_results/mappo_legacy_960k/`. Development candidates stay under
`results/`; none of their final tests are used for tuning. The original reward,
actions, physics and evaluation horizon are unchanged. No download, internet access
or multi-agent training is needed for the recording.

Simple Spread uses `N=3`, discrete actions, a 60-step horizon, and `local_ratio=0.0`.
Every agent receives exactly the same negative sum of nearest-agent distances per
landmark. There are no individual reward bonuses or collision penalties; physical
collisions still affect movement. Team return sums one shared reward over time,
without multiplying by the number of agents. The provided NumPy reward is compared
against actual environment rewards; students can complete a separate practice
version afterward. A shared reward does not by itself reveal whose action helped.

The highway experiment keeps observations (25 features), actions (five), environment,
reward, exploration endpoints and interaction budgets matched. Medium bins are chosen
in advance as the main tabular comparison; coarse/fine tables check discretization
sensitivity. SB3 settings are fixed before collection: a 64–64 MLP, learning rate
0.001, discount 0.95, replay capacity 20,000, warm-up 256, batch size 64,
one optimizer update every four interactions, and target updates every 400 interactions.
Epsilon falls from 1 to 0.05 during the first 70% of the run. The verified tabular
and fixed-baseline records are reused from the earlier study on the same protocol.
Tabular alpha 0.3 came from its development-only pilot; SB3 had no new pilot.
This comparison therefore does not claim equal hyperparameter-search effort.

The full study uses training seeds 0–4. Every 1,000 interactions, policies act without
exploration on the same 20 development traffic seeds (82000–82019). The final
50,000-step policy from each run is evaluated on 100 new test traffic seeds
(93000–93099). Pilot traffic (81000–81019) is separate. Evaluation does not update
weights, insert table entries, or perturb the training RNG. Random actions and
maintaining lane/speed are evaluated once on the same development/test traffic.
Final models are selected by the fixed interaction budget, never by their test score.
SB3 intermediate checkpoints precede the associated optimizer update; its final
checkpoint follows the last update. Its replay buffer preserves bootstrapping on truncation.

The first figure separates **training returns** from **greedy evaluation**. Within
each run, the training line uses the last 50 completed episodes, and the evaluation
line uses the last 5 checkpoint means. Each window updates with every new episode
or checkpoint and uses the data available so far when the window is still filling.
The plotted lines average these rolling means across runs; the evaluation band shows
±1 SD across the smoothed runs, not a confidence interval. Short dips can look
smaller after smoothing. Set a window to `1` in `highway_plot_windows` to see its
unsmoothed trend. Final scores continue to use the original test episodes.

The second figure compares final reward, crash rate and episode duration across
all six methods. Dots are per-training-run means over the same 100 new scenarios,
not individual episodes. The table reports training-only CPU/elapsed time, stored
table states and test-state lookup coverage. Concurrent elapsed times depend on
system load; equal interactions do not equal compute. N/A coverage for DQN does
not imply perfect generalization. The results establish behavior in this one setup,
not universal algorithm superiority. The supplied 20,000-step DQN remains a separate
labelled playback model; it is not in the matched-budget ranking. The study playback
uses seed 0, declared in advance rather than chosen by score.

## Files and generated results

```text
notebooks/                 Student workbook (local instructor source is Git-ignored)
rl_lab/                    Provided simulators, plotting, controls, exercise checks
assets/figures/             Four line-art game overviews (PNG) and generation prompts
assets/checkpoints/         Supplied DQN and a locally trained café table
assets/reference_results/   Measured teaching runs, evaluation CSVs and provenance
vendor/                    Unchanged upstream social games, MIT license and hashes
scripts/                   Generation, environment checks and notebook verification
tests/                     Learning, accounting, rendering and exercise checks
results/                   New runs, figures, GIFs and models (Git-ignored)
```

Each notebook run creates a new results directory and records whether comparison
data were freshly trained or cached. `.npz` checkpoints are loaded with
`allow_pickle=False`. Notebook paths are relative to the checkout, so opening from
the repository root or `notebooks/` works. The supplied lecture PDFs are not copied.

## Maintain and verify

For a student checkout, check the environment and run the available tests:

```bash
python scripts/check_environment.py
python -m pytest -q
```

Tests requiring instructor solutions are skipped when the local instructor notebook
is absent. Student workbook, environment, cached results and library checks still run.

The following authoring and recording commands require the local instructor files,
which are excluded from Git. Keep both notebooks and `instructor_script.md` synchronized
for future lesson revisions, including narration, screen/run cues and results.

Edit **only `notebooks/chapter3_instructor.ipynb`** for shared teaching content.
Markdown cells use `student_source` metadata for detailed student explanations;
exercise cells use it for the TODO scaffold. Recording scaffold previews are tagged
`instructor-only`. Instructor solution cells are visible by default; the generator expands
student exercise cells and strips solution metadata. After editing, regenerate the
student workbook. If you change an exercise interface, also update its scaffold
metadata and check. Do not distribute a student workbook with saved answer outputs.

```bash
python scripts/build_student.py
python scripts/check_environment.py
python -m pytest -q
python scripts/verify_notebooks.py --mode smoke
```

The last command executes the instructor notebook and a temporarily completed
student notebook in separate fresh kernels, including short highway training.
It injects instructor exercise definitions **in memory for verification only**;
it never writes answers into the distributed student notebook. The temporary student
check skips widget display to avoid an nbclient widget-state issue; pytest exercises
the control callbacks, and the instructor check renders all three widget views. Headless widget
snapshots are captured without sending them back into the kernel, avoiding output
feedback loops; live JupyterLab controls remain enabled. The saved instructor file
uses ordinary text/image snapshots with a rerun hint instead of widget-view MIME
outputs, so notebook previews do not show widget rendering errors. Run the control
cells in JupyterLab to use their buttons and sliders. Each results figure has a
one-sentence reading guide. Executed verification
copies and the report go to ignored `results/verification/`.

To rehearse the default presentation and refresh the instructor's saved outputs:

```bash
python scripts/verify_notebooks.py --mode teaching --cached-highway --save-instructor
python scripts/build_student.py
```

To check the recording route and result export without running the self-study
appendix, use `python scripts/verify_notebooks.py --mode teaching --cached-highway --core-only`.
This verifies both editions in fresh kernels, with five charts and two live
instructor controls. The full check also runs Appendix A.4 and its payoff chart
and Dilemma control. Use the full check when refreshing saved instructor previews.

The game overviews are supplied PNG artwork generated with the built-in image tool
using the reference style provided for this lab. The exact prompts and colour rules
are in [assets/figures/prompts.json](assets/figures/prompts.json). They need no image
service or API key when students run the notebook. The café overview's demand curve
is a qualitative schematic; the notebook's analytical plots use simulator data.
The old Matplotlib illustration generator and its SVG exports have been retired.

To run or resume the SB3 experiment on the prepared 8-core Mac:

```bash
python scripts/run_sb3_study.py --workers 5
```

This collects five genuine SB3 runs and reuses the verified tabular/baseline records.
It preserves the original hand-coded study and records its source manifest hash.
Progress and completed-policy checksums are saved under the new cache. Finished
policies with matching protocol and code are reused; an unfinished policy restarts.
Five single-threaded workers leave room for the desktop; use fewer on smaller machines.

The recording cache is in `assets/reference_results/highway_sb3_study/`. Its manifest
identifies the actual DQN implementation, settings, versions, provenance, raw CSVs
and final SB3 `.zip` models. Final tabular models are also included. The notebook
exports raw records and both sets of rolling means under its new `results/` directory.
Only a complete, verified SB3 cache is accepted by the notebook loader.

To reproduce the historical hand-coded experiment:

```bash
python scripts/run_highway_study.py --phase all --workers 6
```

The pilot is followed by the full study. For separate stages, use `--phase pilot`
and then `--phase full`. Six single-threaded CPU workers leave room for the desktop.
Use fewer workers on a smaller machine. The command saves progress every evaluation
checkpoint under ignored `results/highway_study/`. Finished policies with matching
training definitions, settings and checksums are reused; an interrupted **unfinished
policy restarts**. This is resumption between policies, not restoration of optimizer
and replay state within an episode. Changed training definitions require a fresh pilot.
The frozen protocol is saved before full-study test evaluation. A complete manifest
is published only after all policies, raw records and final checkpoints are present.

The historical cache is in `assets/reference_results/highway_study/`. Its manifest
records exact seeds, selected hyperparameters, versions, training fingerprint and
file hashes. Raw completed and unfinished episode records, all development episodes,
checkpoint means and final test episodes are separate CSVs. The notebook exports
these tables plus both sets of rolling means under its new `results/` directory.
The earlier 2,000-step files remain historical data at the parent directory.
The notebooks use the SB3 cache; do not combine results from different DQN implementations.

To refresh the café and original reference-checkpoint evaluation as well as the
historical hand-coded highway study, run:

```bash
python scripts/generate_reference_results.py
```

To refresh only the cookie assets while preserving highway measurements:

```bash
python scripts/generate_reference_results.py --cafe-only
```

Generation replaces bundled reference assets; inspect the changes before committing.

## Sources

The café is a synthetic teaching environment migrated from the supplied TA notebook.
Game source is unchanged at commit `f42dfe8516f5d1290d9076e2563e99f2770d46bf`;
see [vendor/PROVENANCE.json](vendor/PROVENANCE.json) and [vendor/LICENSE](vendor/LICENSE).
See [assets/PROVENANCE.json](assets/PROVENANCE.json) for the source notebook checksum.

- [Gymnasium environment API](https://gymnasium.farama.org/api/env/) and
  [termination versus truncation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)
- [HighwayEnv documentation](https://highway-env.farama.org/)
- [PyTorch DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Stable-Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/dqn.html)
- [PettingZoo-Dilemmas upstream](https://github.com/mrochk/pettingzoo-dilemmas)
- [Simple Spread reward and actions](https://mpe2.farama.org/environments/simple_spread/)
- [skrl MAPPO implementation, pinned release](https://github.com/Toni-SM/skrl/blob/1.4.3/skrl/multi_agents/torch/mappo/mappo.py)
- [MAPPO paper](https://arxiv.org/abs/2103.01955)
- [MPE2 0.0.1 package](https://pypi.org/project/mpe2/0.0.1/)
- [Conda environments](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)

If a highway scene or GIF is a black rectangle, restart the kernel and run the
updated notebook from setup. Earlier versions used SDL's dummy driver without
re-enabling HighwayEnv's offscreen drawing, so they saved blank frames. The current
environment checks verify visible image content, and notebook verification checks
that animations contain multiple visible frames.

## Recorded SB3 experiment

The revised notebooks use five actual SB3 DQN models, each trained for 50,000
interactions. Verified tabular and baseline measurements are reused on the same
protocol. Each final trained policy is tested on the same 100 new traffic scenarios;
each fixed baseline is evaluated once. SB3 settings were fixed without a new pilot.

| Method | Mean test reward | Test crashes | Duration (s) |
|---|---:|---:|---:|
| Tabular · coarse | 11.22 | 76.4% | 14.9 |
| Tabular · medium | 8.59 | 92.8% | 11.4 |
| Tabular · fine | 8.31 | 96.8% | 11.1 |
| DQN · Stable-Baselines3 | 17.87 | 62.2% | 21.6 |
| Random actions | 8.67 | 94.0% | 11.5 |
| Maintain lane / speed | 10.82 | 97.0% | 13.9 |

SB3 improves mean reward over the tabular agents here, but its crash rate remains
high and performance varies across training seeds. Reusing an implementation still
requires careful evaluation. This one setup does not establish a general algorithm ranking.

Raw measurements, final models and provenance are in
`assets/reference_results/highway_sb3_study/`. The earlier hand-coded DQN study
remains in `assets/reference_results/highway_study/`; its DQN scores belong to a
different implementation and are not reused as SB3 results.
