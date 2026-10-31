"""Read instructor definitions for tests/reference generation, never from student runtime."""
import ast
import copy
from pathlib import Path
import nbformat
import numpy as np
import torch
from torch import nn
from rl_lab.highway import random_argmax
from rl_lab.rules import LearningRules

ROOT=Path(__file__).resolve().parents[1]


def load():
    namespace=dict(np=np,torch=torch,nn=nn,copy=copy,random_argmax=random_argmax)
    wanted={'epsilon_greedy','td_target','q_update','dqn_targets','dqn_loss',
            'QNetwork','ReplayBuffer','DQNAgent','tit_for_tat','team_reward'}
    notebook=nbformat.read(ROOT/'notebooks/chapter3_instructor.ipynb',as_version=4)
    for cell in notebook.cells:
        if cell.cell_type!='code':continue
        for node in ast.parse(cell.source).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in wanted:
                exec(compile(ast.Module(body=[node],type_ignores=[]),'<instructor definitions>','exec'),namespace)
    namespace['rules']=LearningRules(*(namespace[name] for name in ['epsilon_greedy','td_target','q_update']))
    return namespace
