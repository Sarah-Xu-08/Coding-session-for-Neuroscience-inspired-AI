#!/bin/zsh
set -eu
TASK_DIR="${0:A:h}"
cd "$TASK_DIR"
if [[ ! -x "$TASK_DIR/.conda/bin/python" ]]; then
  print 'Create the local Conda environment first; see README.md.'
  read '?Press Return to close.'
  exit 1
fi
exec "$TASK_DIR/.conda/bin/python" -m jupyterlab notebooks/chapter3_instructor.ipynb --ServerApp.root_dir="$TASK_DIR"
