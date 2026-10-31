# Recording guide · Chapter 3

Aim for 55–60 minutes including short prediction pauses. Students are learning
NumPy and PyTorch, so narrate shapes and indexing before introducing the full DQN.
Use the instructor notebook; share the student workbook for active practice.

## Rehearse before recording

1. Double-click `start_lab.command`, choose **Python (RL Chapter 3)**, and restart
   the kernel. The first cell should show this checkout's `.conda/bin/python`.
2. Use `RUN_MODE='teaching'` and `RUN_HIGHWAY_TRAINING=False` for a predictable
   presentation. The café still trains; prepare a cut or pause for that cell.
3. Run all instructor cells once. Check the figures, all three widget controls,
   saved GIFs, and the run metadata under `results/`.
4. Reset the café to seed 2026 before demonstrating decisions. Preparation arrives
   next hour. Read the inventory before choosing; reveal demand afterwards.
5. Explain that highway comparison data are recorded. The 20,000-step reference
   animation is a separate model with a larger budget. The notebook labels both.
6. Increase browser zoom if necessary; collapse lengthy replay-buffer code while
   keeping the network, target/loss functions, and `learn` method visible.
7. After any edits, regenerate the student notebook. It should contain no saved
   outputs and no embedded answers in its metadata or code.

## Suggested sequence

| Time | What to show | Pause prompt |
|---|---|---|
| 00–04 | Observation → action → reward; four-task agenda | What can each agent observe before acting? |
| 04–08 | NumPy table, row indexing, feasible-action mask | Which array axis represents actions? |
| 08–12 | Tensor batch, linear layer, gradient and optimizer | Does a forward pass update weights? |
| 12–17 | Café rules and two manual decisions | Which dessert should be prepared before its peak? |
| 17–23 | Exercise 1 solutions and checks | What happens to the target at closing? |
| 23–28 | Café results and policy slices | Can reward improve while cash profit falls? |
| 28–33 | Highway scene, 25 features, coarse/fine bins | What does finer binning cost in coverage? |
| 33–41 | Network shapes, exercise 3, replay, target network | Where should gradients stop? |
| 41–46 | Matched-budget comparison and labelled reference GIF | What can this one-seed result support? |
| 46–52 | Prisoner's Dilemma, tit-for-tat, strategy tournament | Who exploits an always-cooperating partner? |
| 52–58 | Simple Spread illustration, NumPy reward and matched controllers | Why can moving away from your nearest landmark help the team? |
| 58–60 | Exit questions and optional experiments | What stays the same from a Q-table to a network? |

For a 40–45 minute video, move the detailed replay-buffer walkthrough, policy-map
slices, and optional independent social learners into follow-up practice. Keep the
array/tensor warm-up and all five exercise prompts. For student pause time beyond
the video, say “pause here, complete the function, then run the check.”

## Answers and common mistakes

**Exploration:** Both exploration and greedy selection must respect the mask. A
mask selects feasible actions, not high values. Equal maxima should be chosen
randomly. Maximum next-state Q may be negative; do not silently replace it with zero.

**TD updates:** At true termination use reward only. For an external time limit,
stop the rollout but retain the bootstrap term. The café horizon is part of the
task with time observed; the highway cutoff is treated as truncation. Show the
hand calculation `1 + .25 * (3 - 1) = 1.5`.

**Café interpretation:** Inventory prepared now becomes available next hour. The
forecast baseline knows the demand curve. Gray policy cells mean unvisited, and
some are unreachable; they do not indicate a learned “wait” action. Compare cash
profit, waste, and missed requests separately. A new reward function requires
retraining before making policy claims.

**PyTorch:** A state batch is `(B,25)`, network output `(B,5)`, selected values
`(B,)`, targets `(B,)`, and mean loss scalar. Action indices use integer dtype.
Avoid broadcasting `(B,1)` against `(B,)`, which can produce a `(B,B)` loss array.
`zero_grad`, `backward`, and `step` have distinct roles. The target network is
copied periodically; it receives no optimizer updates through the target.

**Highway:** Fine bins can produce many unseen evaluation keys. DQN can produce
outputs at unseen inputs, but those values are not automatically good. Equal
interactions do not equalize compute or learning updates. Cached teaching results
are real measurements, with one seed and a deliberately short training budget.
The supplied 20,000-step checkpoint has a different budget and must remain separate.
Do not claim DQN must win. Always inspect collisions alongside return.

**Prisoner's Dilemma:** Always defect gets 5 against always cooperate; mutual
cooperation gives each 3. Tit-for-tat cooperates initially, then responds to the
previous action. It is hand-coded, not trained, and does not dominate all strategies.
Simultaneous choices cannot inspect the partner's current-round action.

**Simple Spread:** Start with the illustration: three agents choosing their own
nearest landmark can leave two landmarks unattended. The reward instead examines
each landmark's nearest agent. In the two-landmark example the start gives −9,
both left gives −10, and splitting exactly across the landmarks gives 0. Narrate
the `(agents, landmarks, coordinates)` broadcast, norm over coordinates, minimum
over agents (`axis=0`), then sum and negate. Every agent receives this same scalar.
`local_ratio=0.0` removes individual collision penalties; collisions still affect
the physics. Demonstrate the actual reward dictionary before comparing controllers.
Use the selector and Play/step controls to inspect real simulator trajectories.
The assignment controller uses centralized global information and is hand-coded;
do not present it as learned or decentralized coordination. Team return sums the
shared reward over time once. An agent that stays still can benefit from another
agent's move; this illustrates the credit-assignment problem. Shared reward does
not require identical observations, actions, or network weights.

**Optional social RL:** The state includes remaining rounds and the last joint
action. With two learners the opponent changes during training. Fixed-horizon
independent learning does not guarantee cooperation or convergence.

## Maintain the teaching files

The instructor notebook is the source. Instructor-only cells are tagged;
exercise TODO alternatives live in `student_source` metadata. Run
`python scripts/build_student.py` after editing. Keep algorithm code in the
notebooks and environment/plot bookkeeping in `rl_lab/`.

The README lists checks and generation commands. Longer training is optional;
recorded outputs should always identify their budget, seed, versions, and source.
Google Colab support is deferred until the local lesson is settled.
