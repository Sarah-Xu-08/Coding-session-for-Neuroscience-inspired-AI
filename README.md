# Neuroscience-inspired AI · Chapter 3 coding lab

A hands-on reinforcement learning session covering Q-learning in a cookie café,
DQN and Stable-Baselines3 for highway driving, and multi-agent coordination with
MAPPO. Prisoner's Dilemma is an optional appendix activity.

Open the **[student notebook](notebooks/chapter3_student.ipynb)** and follow the
video through Tasks 1–3. Detailed explanations are in the notebook.

## Set up locally: macOS, Windows or Linux

Install [Miniconda](https://www.anaconda.com/docs/getting-started/installation)
(or Anaconda) and Git. Then clone the repository:

```bash
git clone https://github.com/Sarah-Xu-08/Coding-session-for-Neuroscience-inspired-AI.git
cd Coding-session-for-Neuroscience-inspired-AI
```

**macOS / Linux — Terminal:**

```bash
conda env create --prefix ./.conda --file environment.yml
conda activate ./.conda
```

**Windows — Anaconda / Miniconda Prompt:**

```bat
conda env create --prefix .\.conda --file environment.yml
conda activate .\.conda
```

**Then, on any platform:**

```bash
python -m ipykernel install --user --name rl-chapter3 --display-name "Python (RL Chapter 3)"
python scripts/check_environment.py
python -m jupyterlab notebooks/chapter3_student.ipynb
```

Select **Python (RL Chapter 3)** as the notebook kernel. Repeat the activation
command whenever you open a new terminal. No GPU is required.

[environment.yml](environment.yml) specifies Python 3.11 and the lesson's direct
dependencies. Use it for setup; `requirements-macos-arm64.lock.txt` records the
original Mac installation and is not a Windows/Linux requirements file.
The setup has been tested on Apple Silicon macOS; Windows and Linux have not yet
been validated. Run the environment check above on your machine.

## Follow the lesson

Run cells in order. Fill each task's marked `...`, run the function cell, then run
its check. Unfinished tasks intentionally raise `NotImplementedError`; use Run All
after completing Tasks 1–3. Appendix exercises are optional.

Prepared highway results and multi-agent policies are included, so long training
runs are unnecessary. Interactive buttons and sliders need a running notebook;
they do not work in GitHub's preview. New outputs are saved under `results/`.

## Google Colab option

You can also use [Google Colab](https://colab.research.google.com/) if you customize
the environment. Open the student notebook, clone or upload the **full repository**,
and set the working directory to its root so the code and assets can be found.

Add setup cells based on the pip dependencies in [environment.yml](environment.yml).
Check the runtime's Python version: the notebook requires **Python 3.11**, so another
version needs a matching runtime or verified changes to the code and dependencies.
Adapt package versions and widget setup as needed, and restart the runtime after
changing installed packages. See [Colab's runtime guidance](https://research.google.com/colaboratory/runtime-version-faq.html).

## Further details

- [Validation and experiment notes](VALIDATION.md)
- [Asset provenance](assets/PROVENANCE.json)
- [Bundled environment provenance](vendor/PROVENANCE.json) and [license](vendor/LICENSE)
