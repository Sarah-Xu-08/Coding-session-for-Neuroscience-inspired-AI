#!/bin/zsh
set -eu
TASK_DIR="${0:A:h}"
cd "$TASK_DIR"
if [[ ! -x "$TASK_DIR/.conda/bin/python" ]]; then
  print 'Create the local Conda environment first; see README.md.'
  read '?Press Return to close.'
  exit 1
fi
NOTEBOOK="notebooks/chapter3_student.ipynb"
if [[ -f "$TASK_DIR/notebooks/chapter3_instructor.ipynb" ]]; then
  NOTEBOOK="notebooks/chapter3_instructor.ipynb"
fi
exec "$TASK_DIR/.conda/bin/python" -m jupyterlab "$NOTEBOOK" --ServerApp.root_dir="$TASK_DIR"
