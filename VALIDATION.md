# Local validation · 2026-11-16

Validated on the recording machine: Apple Silicon (`arm64`), macOS, Python 3.11,
CPU PyTorch 2.5.1. Direct dependencies are in `environment.yml`; the installed pip
versions are recorded in `requirements-macos-arm64.lock.txt`.

| Check | Result |
|---|---|
| Highway visible output | Offscreen scene and GIF contain visible road/vehicles; animation has multiple different frames; brown controlled car and grey traffic verified |
| Highway rendering independence | Seeded observations, rewards, termination and speed match with and without rendering |
| Highway comparison | Full study: 20 policies, 50,000 interactions each, five seeds per method; 20 development scenarios every 1,000 steps, 100 disjoint final scenarios; fixed baselines; all raw records and final model hashes verified |
| Actual SB3 cache | Five new `.zip` models, each with 50,000 interactions and 12,436 optimizer updates; verified tabular/baseline records reused with source hashes; loader rejects historical hand-coded DQN results |
| Saved-model agreement | Reloaded SB3 seed-0 model reproduces every metric in all 100 recorded final test episodes |
| All four environments | Passed, including HighwayEnv RGB rendering and checkpoint loading |
| Automated behavior checks | 59 passed on 2026-11-17, including actual SB3 training, time-limit bootstrapping, evaluation isolation, model save/load, notebook backend selection, exact scaffolds and optional-exercise isolation |
| Student Git checkout | On 2026-11-17, an isolated copy of the staged files passed 39 tests; 20 tests requiring local instructor materials were skipped. Environment checks passed for all four environments, rendering and supplied checkpoints. The student notebook has no saved outputs or solution metadata. |
| Café demand schedule | Two unequal mean-demand peaks at 12:00 and 18:00; selling hours 10:00–19:00, closing at 20:00; evening simulation checked |
| Café model-optimal reference | Analytic zero-demand and final-hour cases passed; exact Poisson inventory probabilities and expected rewards agree with actual customer simulations; optimal expected daily reward agrees with independent simulator rollouts |
| Café plot semantics | One brown cookie matches the introduction; real-day recommendations use the same observed states; full hour × stock tables have no hidden resource slice; ? means no training update |
| Fresh-kernel smoke execution | Instructor and temporarily completed student flows passed; includes new short highway training |
| Default teaching execution | Instructor and temporarily completed student flows passed in fresh kernels |
| Instructor presentation outputs | Live run: 6 inline PNG charts, 1 GIF and 3 widget views; saved preview: 7 PNGs (including the captured Spread frame), 1 GIF, and readable control hints |
| Student teaching outputs during verification | 6 inline PNG charts and 1 GIF; widget display skipped only in the temporary headless check |
| Headless widget capture | Instructor snapshots retain café text and the Simple Spread image without kernel feedback; live controls remain enabled |
| Portable saved preview | No widget-view MIME or widget state remains in the saved instructor file; nested text/images are retained, missing state gets a rerun hint, and live cell source is preserved |
| Control callbacks | Café advance/close/reset, dilemma strategy selection/play, and Simple Spread strategy/step controls passed |
| Reliable preparation and state | Passed: state (hour, stock), Q shape (11, 9, 3); every valid order arrives next hour, never serves current customers; demand is the only uncertainty |
| Distributed student workbook | 36 code cells; no saved outputs, instructor-only cells, or solution metadata |
| Problem-first teaching flow | Café decisions precede NumPy Q-table indexing; highway observations precede PyTorch tensors and a real-transition DQN update, followed immediately by SB3 and the experiment |
| Upstream games | All bundled source hashes match the supplied manifest |
| Simple Spread shared rewards | All three agents' actual rewards match the NumPy coverage formula, including physical overlaps, on seeds 0, 7 and 2026 |
| Simple Spread baselines | Same initial world and 60-step budget; no learning; centralized assignment is labelled explicitly |
| Introductory illustrations | Four selective-colour line-art PNGs, visually reviewed against the supplied reference; one at each game's introduction in both notebooks |
| 30-minute recording flow | Matching opening roadmap with brief Chapter 3 connections and no on-screen timetable; three required tasks; concise instructor prose and detailed student reference; optional activities follow the recording finish point |
| Stable-Baselines3 | 2.7.1 installed and pinned; introduced immediately after the hand-coded update, then used for the experiment and playback; default API demonstration is explicitly untrained |
| Multi-agent purpose | Trained Independent Q-learning and skrl MAPPO are distinguished from fixed tit-for-tat, nearest and assignment rules; shared reward retains the credit-assignment question |
| Dependency consistency | `pip check` passed; MPE2 0.0.1 works with the existing pygame-based Mac environment |

The maintained instructor notebook includes the default teaching outputs. The
distributed student notebook remains unfinished by design. Verification injects
solutions into a temporary in-memory copy; it never fills the distributed file.
Live controls remain enabled in the local instructor notebook and distributed
student notebook. Student widget display
is skipped in the headless check because of an nbclient widget-state issue; control
callbacks and instructor widget views are checked separately.

The bundled café reference was trained locally for 60,000 days, seed 0. The default SB3
highway study contains 1,000,000 total training interactions: four learned methods
across five seeds with 50,000 interactions per policy. Five new SB3 DQN runs supply
250,000 of these interactions; verified tabular/baseline records are reused from
the historical study. Tabular alpha 0.3 came from its development-only pilot.
SB3 settings were fixed before collection without an SB3 pilot; equal tuning effort
is not claimed. There are
20,400 full-study development episodes and 2,000 final test episodes, plus fixed
baselines on the same traffic. Completed and partial training records account for
every interaction. The learning figure uses rolling means over the last 50 completed
training episodes and the last 5 evaluation checkpoints within each run. The
evaluation band shows ±1 SD across smoothed runs, not a confidence interval.
Training traffic reset streams were audited against reserved development/test seeds.
`assets/reference_results/highway_sb3_study/manifest.json` records the actual
implementation, provenance, hashes and final SB3 `.zip` models. The hand-coded
`highway_study/` and original short-run files remain historical data.

The supplied 20,000-step DQN checkpoint was re-evaluated locally on seeds
92000–92019. Its mean return was 19.415 and collision fraction 0.10, matching the
original recorded evaluation. See `assets/reference_results/manifest.json` and
the per-episode CSV for the measured results. This checkpoint remains separate
from the matched 50,000-step comparison.

MPE2 0.0.1 supplies the actual Simple Spread simulator. `local_ratio=0.0` selects
only the shared global coverage reward. Individual collision penalties are absent;
physical collisions remain enabled. Nearest and assignment are fixed global-position
references. Library MAPPO uses local actors and global-state training critics; its
actual learned policy is evaluated separately from those references.
The old overview and Stag Hunt teaching figures have been removed.

SciPy 1.14.1 loads and runs on this Mac; the initially tried 1.15.3 wheel did not.
The notebook explicitly enables inline Matplotlib plotting. The environment check,
tests, and verification commands are documented in the README. Google Colab and
Windows have not been validated in this release.

The historical hand-coded study's pilot used eight 10,000-interaction runs on
development-only traffic. Both studies use final-budget policies rather than
choosing by test score; study playback uses seed 0 fixed in advance. Training
CPU/elapsed times exclude evaluation, with concurrent elapsed time depending on
system load. Re-running either offline command reuses matching completed policies;
unfinished policies restart rather than restoring optimizer/replay state.

Earlier 2026-10-31 execution timings: teaching instructor 26.3 seconds, student 20.4 seconds; smoke instructor 85.9 seconds, student 87.6 seconds.

2026-11-15 revised-flow checks: fresh-kernel teaching instructor 15.39 seconds and student 11.55 seconds; fresh-kernel live-training smoke instructor 23.97 seconds and student 21.65 seconds. Instructor teaching outputs were refreshed again in 12.28 seconds after aligning the rolling-average reference text. Student answers were injected only into temporary verification copies.

Final plot review: shared reward scale, unobstructed legend, five visible run markers and matching brown verified. All 28 completed pilot/full journals were copied to the prepared Desktop checkout and verified for resumption.

2026-11-15 rolling mean update: all seven highway study tests passed. The notebook's
comparison cell rendered both windows from the verified cached study and exported
the rolling values. The final policy comparison image matched the earlier version.

2026-11-15 SB3 experiment revision: all 46 behavior tests and environment checks
passed. Fresh-kernel live-training smoke runs passed for both notebooks (139.53 s
instructor, 175.01 s student, while the full five-worker collection was active).
With the completed SB3 cache, fresh-kernel teaching runs passed in 25.11 s and
24.76 s. The instructor's saved outputs were refreshed; student answers were used
only in temporary verification copies. All five SB3 models and source/data hashes
were verified. Held-out mean reward was 17.87 and crash rate 62.2%; variation among
training seeds remains visible rather than selecting the best run for playback.
Final SB3 learning/outcome figures were visually checked for readable labels,
unobstructed legends, uncertainty bands and all five run markers. Resumption
reused all five completed SB3 policies with training disabled by a verification guard.

2026-11-16 presentation revision: 21 targeted notebook, preview and environment
checks passed. Fresh-kernel cached teaching runs passed for instructor (14.73 s)
and student (12.09 s). Both Section 1 titles were preserved exactly. Section and
subsection titles now lead with learning goals; result figures have one-sentence
reading guides. A separate optional-activity run verified all eight PNG figures,
both GIFs and eleven reading guides/table notes. The maintained instructor file
contains eight figure explanations, captured café text, the Spread scene, and no
widget-view MIME or widget state. Live widget behavior remains checked separately;
the generated student workbook has no saved outputs.


2026-11-16 learned multi-agent revision (historical MAPPO model; superseded below):

- Installed/pinned skrl 1.4.3; `pip check` and environment/model-loading checks pass.
- Independent Q-learning uses two separate Q-tables trained for 3,000 twenty-round
  self-play episodes. Frozen reload reproduces the saved self-play evaluation;
  matches and controls leave Q-values unchanged.
- MAPPO is the actual installed `skrl.multi_agents.torch.mappo.MAPPO` class.
  A short test performs a real library update, verifies changed actor weights,
  reloads its checkpoint, and reproduces a recorded evaluation trajectory.
  The 18-input actors run without calling the 54-input training critics.
- Prepared MAPPO training: 960,000 joint steps (2,880,000 individual actions), seed 0;
  checkpoint selection uses development seeds 71000–71007. The initial 240,000-step
  pilot was extended after incomplete coverage; final test seeds 91000–91019 were
  reserved before the final run and not used for selection.
- Final held-out mean team returns: MAPPO −77.43, nearest −57.58, assignment −18.98.
  The clustered teaching scene gives MAPPO −111.01 versus nearest −166.00.
  The notebooks and script explain incomplete coverage and the difference between
  that illustration and held-out performance; no claim of reliable coordination.
- All 52 automated tests pass, including learned-policy control callbacks,
  frozen evaluation, shared rewards, checkpoint hash rejection and existing lessons.
- Both revised notebooks pass default teaching execution in fresh kernels. The
  instructor has 39 executable cells, the student 36. Student solutions are injected
  only into a temporary verification copy. The distributed workbook remains blank
  at the three required tasks and two optional exercises, with no outputs/solutions.
- All 58 main notebook cell IDs match the script's recording stops in order.
  Spoken narration remains about 3,100 words, with the roughly 30-minute route kept
  in recording documents. The final MAPPO plot was visually checked: common world
  coordinates retain the drifting agent, fixed-size position markers remain readable,
  and the reward legend is below the curve panel.


2026-11-16 MAPPO quality revision:

- Kept the genuine skrl 1.4.3 learner. The selected model shares actor/critic
  parameters across interchangeable agents; actor inputs remain local. Six
  distances derived from those local observations give 24 actor inputs; corresponding
  features give 72 critic inputs from the raw 54-value global state. No assigned
  targets, expert actions or hand-coded action fallback enter the learned policy.
- Training uses library time-limit bootstrapping, LayerNorm networks, constant
  reward scaling (0.1) and an unclipped critic via a wide clipping window for
  skrl 1.4.3 compatibility. Environment reward, physics, actions and 60-step
  evaluation horizon are unchanged; all reported scores are in original units.
- Three configurations each trained for 3,000,000 joint steps (9,000,000 total
  development interactions). All use training seed 0. The selected configuration's
  best development checkpoint is at 2,400,960 steps; its final training checkpoint
  is kept separately and is not used for recording. Settings, all development
  curves and integration source are archived with hashes.
- Training starts are 80% random and 20% the practiced clustered teaching scene.
  Reset seeds lie in [1,000,000,000, 2,000,000,000). Development seeds 171000–171015
  select configurations/checkpoints. Final seeds 271000–271049 remained reserved
  until selection was fixed. The runner rejects further training after final
  evaluation. This is one training seed, not a multi-seed benchmark.
- The selected model's development mean return is −26.09 and final coverage 3/3.
  The recording scene covers all three landmarks, with team return −26.24 and
  final reward −0.108. Readiness checks require actual coverage and no drifting
  away, rather than merely successful execution.
- On 50 matched new random starts, mean return improves from −86.22 for the
  archived previous model to −33.90; mean final landmarks covered improves from
  0.64 to 2.70/3. Coverage counts landmarks with an agent within 0.2 world units
  at the final step. Improved MAPPO covers all three in 41/50 starts. This
  diagnostic is not a reward modification. Nearest steering scores −64.73 and
  centralized assignment −21.29; the fixed planner still performs better.
- The archived previous model was compared on the same 50 final starts only
  after new-model selection was frozen. Its original checkpoint, original
  integration source and original evaluation remain under
  `assets/reference_results/mappo_legacy_960k/`.
- All 56 automated tests pass. Tests perform real short library training and
  resumption, check shared batching and local-only distance features, verify
  bootstrapping without modifying reported rewards, and reload the selected model
  to reproduce all 50 recorded final evaluations. Quality regression checks require
  improved held-out return over nearest steering, mean coverage at least 2.5/3,
  full teaching-scene coverage and bounded trajectories.
- Fresh-kernel teaching execution passes for instructor (10.85 s; 39 code cells)
  and temporarily completed student (11.02 s; 36 code cells). Instructor
  previews were refreshed from the improved policy, retaining six direct PNGs,
  a captured Spread image, one GIF and portable control hints. The student file
  was regenerated afterward and contains no saved outputs or solution metadata.
  Instructor answers remain visible; all 58 main recording cell IDs match the
  script in order. Both narratives explain the practiced scene, new-start
  evaluation, remaining planner advantage and shared-reward credit assignment.


2026-11-16 compact recording route:

- Moved all six original Prisoner's Dilemma cells to Appendix A.4, retaining
  their IDs and behavior. A.4.1 supplies payoffs/tit-for-tat, A.4.2 compares frozen
  Independent Q-learners, and A.4.3 retains optional Task 4. Conflicting individual
  incentives and possible mutual cooperation are distinguished from the main
  lesson's shared team objective. The main route gives a brief self-study pointer.
- Simple Spread is now Section 3, with reward and MAPPO subsections 3.1/3.2;
  the conclusion is Section 4. Both overviews, transitions, conclusion, private
  timing plan and script run cues match. All 52 main cell IDs match the recording
  stops in order, and all original notebook IDs are retained. The route plans
  about 29 minutes plus room for natural pauses within the 30-minute limit.
- Core result export works without any appendix activity. Optional tournament
  output and metadata are included only after A.4.2 is run. Cached policies,
  experiment manifests and learning code are unchanged.
- All 56 automated tests pass. Full fresh-kernel teaching execution passes for
  instructor (11.15 s; 39 code cells) and temporarily completed
  student (11.76 s; 36 code cells), including the moved appendix.
- The new `--core-only` verification passes independently in fresh kernels for
  instructor (11.31 s) and student (12.29 s). It executes only the
  recording route plus result export, checks that no Dilemma variables were
  created, and verifies null Dilemma metadata/no tournament file. Each route
  produces five charts and one driving GIF; the live instructor has two controls.
- Instructor saved previews were refreshed with the full verifier. The student
  file was regenerated afterward; it remains unanswered with no saved outputs
  or solution metadata. The reward equations and MAPPO/credit-assignment
  explanations remain on the main route.
