"""Provided repeated-run experiment, raw episode records and lightly smoothed plots."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from . import ROOT
from .highway import (AGENT_BROWN, HIGHWAY_CONFIG, SparseTrafficQ, PlaybackDQN,
                      highway_evaluate, train_highway)

METHODS = ('coarse', 'medium', 'fine', 'DQN')
NAMES = {'coarse': 'Tabular · coarse', 'medium': 'Tabular · medium',
         'fine': 'Tabular · fine', 'DQN': 'DQN',
         'random': 'Random actions', 'maintain': 'Maintain lane / speed'}
COLOURS = {'coarse': '#c7c7c7', 'medium': '#626262', 'fine': '#a0a0a0',
           'DQN': AGENT_BROWN, 'random': '#d8d8d8', 'maintain': '#b9b9b9'}
TABLE_FILES = {'training': 'training_episodes.csv', 'evaluation': 'development_episodes.csv',
               'curves': 'evaluation_checkpoints.csv', 'scores': 'test_episodes.csv',
               'baselines': 'baseline_episodes.csv'}


def method_name(kind, study):
    if kind == 'DQN' and study.metadata.get('dqn_implementation') == 'stable_baselines3.DQN':
        return 'DQN · Stable-Baselines3'
    return NAMES[kind]


@dataclass
class HighwayStudy:
    training: pd.DataFrame
    evaluation: pd.DataFrame
    curves: pd.DataFrame
    scores: pd.DataFrame
    baselines: pd.DataFrame
    metadata: dict
    agents: dict = field(default_factory=dict)


class FixedPolicy:
    """Non-learning baselines; use only the same available observation interface."""
    def __init__(self, kind, seed=0):
        if kind not in {'random', 'maintain'}:raise ValueError(kind)
        self.kind, self.rng = kind, np.random.default_rng(seed)

    def act(self, state):
        return int(self.rng.integers(5)) if self.kind == 'random' else 1


def baseline_evaluation(development_seeds, test_seeds):
    rows=[]
    for kind in ['random', 'maintain']:
        for split, seeds in [('development', development_seeds), ('test', test_seeds)]:
            if not len(seeds):continue
            frame=highway_evaluate(FixedPolicy(kind), seeds)
            rows.append(frame.assign(method=kind, split=split, train_seed=-1,
                                     train_seconds=0.,train_cpu_seconds=0.,table_states=np.nan))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_highway_study(rules, dqn_factory, steps=50000, seeds=range(5), eval_every=1000,
                     eval_seeds=range(82000,82020), test_seeds=range(93000,93100),
                     methods=METHODS, dqn_kwargs=None, table_kwargs=None, device='cpu'):
    """Sequential classroom runner. The recording experiment uses parallel workers."""
    seeds,eval_seeds,test_seeds=list(seeds),list(eval_seeds),list(test_seeds)
    if set(eval_seeds)&set(test_seeds):raise ValueError('Development and final test traffic must be disjoint.')
    agents={};training=[];evaluation=[];curves=[];scores=[]
    for seed in seeds:
        for kind in methods:
            agent,curve=train_highway(kind,rules,dqn_factory,steps=steps,seed=seed,
                                      eval_every=eval_every,eval_seeds=eval_seeds,device=device,
                                      agent_kwargs=dqn_kwargs if kind=='DQN' else table_kwargs)
            agents[(kind,seed)]=agent
            training.append(agent.training_episodes);evaluation.append(agent.development_episodes)
            curves.append(curve.assign(method=kind,seed=seed))
            frame=highway_evaluate(agent,test_seeds)
            scores.append(frame.assign(method=kind,train_seed=seed,
                train_seconds=agent.training_seconds,train_cpu_seconds=agent.training_cpu_seconds,
                table_states=len(agent.q) if isinstance(agent,SparseTrafficQ) else np.nan))
    metadata=dict(experiment_id='highway_episode_study_v1',steps_per_method=steps,
                  training_seeds=seeds,development_seeds=eval_seeds,test_seeds=test_seeds,
                  eval_every=eval_every,methods=list(methods),
                  dqn_kwargs=dqn_kwargs or {},table_kwargs=table_kwargs or {},
                  source='trained in this notebook',config=HIGHWAY_CONFIG,complete=True)
    return HighwayStudy(*(pd.concat(part,ignore_index=True) for part in
                          [training,evaluation,curves,scores]),
                        baseline_evaluation(eval_seeds,test_seeds),metadata,agents)


def save_study(study, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for key,name in TABLE_FILES.items():getattr(study,key).to_csv(folder/name,index=False)
    metadata=dict(study.metadata)
    metadata['files_sha256']={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                              for name in TABLE_FILES.values()}
    (folder/'manifest.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')


def validate_study(study):
    """Check that published data implement their declared interaction/seed budget."""
    metadata=study.metadata
    if set(metadata['development_seeds'])&set(metadata['test_seeds']):
        raise ValueError('Development and test traffic overlap.')
    expected={(method,seed) for method in metadata['methods'] for seed in metadata['training_seeds']}
    training=study.training.groupby(['method','train_seed'])
    if set(training.groups)!=expected:raise ValueError('Missing or unexpected training runs.')
    if not (training.length.sum()==metadata['steps_per_method']).all():
        raise ValueError('Training episode records do not account for the declared budget.')
    steps=metadata['steps_per_method'];interval=metadata['eval_every']
    checkpoints={0,*range(interval,steps+1,interval),steps}
    for run in expected:
        method,seed=run
        curve=study.curves[(study.curves.method==method)&(study.curves.seed==seed)]
        if set(curve.step)!=checkpoints or len(curve)!=len(checkpoints):
            raise ValueError(f'Missing or duplicated evaluation checkpoints: {run}')
        development=study.evaluation[(study.evaluation.method==method)&(study.evaluation.train_seed==seed)]
        for step in checkpoints:
            part=development[development.step==step]
            if len(part)!=len(metadata['development_seeds']) or set(part.seed)!=set(metadata['development_seeds']):
                raise ValueError(f'Wrong development traffic: {run}, step {step}')
            mean=curve.loc[curve.step==step,'return_'].iloc[0]
            if not np.isclose(mean,part.return_.mean()):raise ValueError('Checkpoint mean disagrees with episode data.')
        test=study.scores[(study.scores.method==method)&(study.scores.train_seed==seed)]
        if len(test)!=len(metadata['test_seeds']) or set(test.seed)!=set(metadata['test_seeds']):
            raise ValueError(f'Wrong final test traffic: {run}')
    return study


def load_cached_study(folder=None):
    folder=Path(folder or ROOT/'assets/reference_results/highway_study')
    metadata=json.loads((folder/'manifest.json').read_text())
    if not metadata.get('complete'):raise RuntimeError('The highway study has not finished.')
    for name,expected in metadata.get('files_sha256',{}).items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Highway study checksum mismatch: {name}')
    tables={key:pd.read_csv(folder/name) for key,name in TABLE_FILES.items()}
    return validate_study(HighwayStudy(**tables,metadata=metadata))


def load_study_dqn(seed=0, folder=None):
    folder=Path(folder or ROOT/'assets/reference_results/highway_study')
    return PlaybackDQN(folder/'runs'/f'DQN_seed{seed}'/'agent.npz')


def _rolling_rewards(frame, seed_column, window):
    """Use the current and previous returns within each run, in time order."""
    if not pd.api.types.is_integer(window) or window < 1:
        raise ValueError('The rolling window must be a positive integer.')
    group = ['method', seed_column]
    ordered = frame.sort_values([*group, 'step']).copy()
    ordered['rolling_mean'] = ordered.groupby(group, sort=False)['return_'].transform(
        lambda values: values.rolling(window=window, min_periods=1).mean())
    return ordered


def training_rolling_mean(training, window):
    """Smooth completed training episodes; an unfinished episode has no final return."""
    return _rolling_rewards(training.loc[training.complete.astype(bool)], 'train_seed', window)


def evaluation_rolling_mean(curves, window):
    """Smooth the recorded checkpoint means separately for each training run."""
    return _rolling_rewards(curves, 'seed', window)


def training_trace_summary(frame):
    """Align runs at every observed episode endpoint using each run's LAST estimate.

    Forward filling carries an existing estimate; it does not average time bins
    or interpolate new episode returns. Start after all runs have one episode.
    """
    traces=[part.set_index('step')['rolling_mean'] for _,part in frame.groupby('train_seed')]
    if not traces:return pd.DataFrame(columns=['mean','std'])
    endpoints=np.unique(np.concatenate([trace.index.to_numpy() for trace in traces]))
    aligned=pd.concat([trace.reindex(endpoints).ffill() for trace in traces],axis=1)
    aligned=aligned.dropna()
    return pd.DataFrame({'mean':aligned.mean(axis=1),'std':aligned.std(axis=1,ddof=0)})


def plot_highway_learning(study, save_dir=None, *, training_window=50, evaluation_window=5):
    """Plot both rolling trends and return the smoothed per-run records for export."""
    training = training_rolling_mean(study.training, training_window)
    evaluation = evaluation_rolling_mean(study.curves, evaluation_window)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),sharey=True,layout='constrained')
    primary=[kind for kind in ['medium','DQN'] if kind in set(study.curves.method)]
    if not primary:primary=[kind for kind in METHODS if kind in set(study.curves.method)]
    for kind in primary:
        colour=COLOURS[kind];part=training[training.method==kind]
        summary=training_trace_summary(part)
        if not summary.empty:
            axes[0].plot(summary.index,summary['mean'],color=colour,lw=2,
                         linestyle='-' if kind=='DQN' else '--',label=method_name(kind,study))
        # Summarize the smoothed checkpoint means, weighting each training run equally.
        runs=evaluation[evaluation.method==kind]
        checkpoints=runs.groupby('step')['rolling_mean'].agg(['mean','std'])
        axes[1].plot(checkpoints.index,checkpoints['mean'],color=colour,lw=2,
                     linestyle='-' if kind=='DQN' else '--',marker='o',markersize=2.5,label=method_name(kind,study))
        if runs.seed.nunique()>1:
            spread=checkpoints['std'].fillna(0)
            axes[1].fill_between(checkpoints.index,checkpoints['mean']-spread,
                                  checkpoints['mean']+spread,color=colour,alpha=.14)
    development=study.baselines[study.baselines.split=='development']
    for kind,pattern in [('random',':'),('maintain','-.')]:
        values=development.loc[development.method==kind,'return_']
        if len(values):axes[1].axhline(values.mean(),color='#929292',ls=pattern,lw=1,label=NAMES[kind])
    axes[0].set(title=f'Training · {training_window}-episode rolling mean',
                 xlabel='Training interactions',ylabel='Episode reward (exploration enabled)')
    axes[1].set(title=f'Evaluation · {evaluation_window}-checkpoint rolling mean',
                 xlabel='Training interactions',ylabel='Mean reward (exploration disabled)')
    for ax in axes:
        ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=9)
    # Use the episode-reward scale rather than crowding the curves and references.
    low,high=axes[1].get_ylim()
    axes[1].set_ylim(min(0,low),max(30,high))
    axes[1].tick_params(axis='y',labelleft=True)
    axes[1].legend(loc='upper right',ncol=2,frameon=False,fontsize=8)
    seeds=len(study.metadata['training_seeds']);n=len(study.metadata['development_seeds'])
    fig.suptitle('Highway · training progress and learned-policy evaluation',fontsize=15,weight='bold')
    fig.supxlabel(f'{seeds} training run(s) per method · {n} evaluation episodes per checkpoint\n'
                  'Trailing windows · band = ±1 SD across smoothed evaluation runs, not a confidence interval.',fontsize=9)
    if save_dir is not None:fig.savefig(Path(save_dir)/'highway_learning_curves.png',dpi=180)
    plt.show()
    return training, evaluation


def study_summary(study):
    baselines=study.baselines[study.baselines.split=='test'].drop(columns='split')
    scores=pd.concat([study.scores,baselines],ignore_index=True)
    per_run=scores.groupby(['method','train_seed']).agg(
        mean_reward=('return_','mean'),crash_rate=('collision','mean'),
        duration_s=('duration_s','mean'),mean_speed_m_s=('mean_speed_m_s','mean'),
        train_cpu_seconds=('train_cpu_seconds','mean'),train_wall_seconds=('train_seconds','mean'),
        table_states=('table_states','mean'),seen_fraction=('seen_fraction','mean'))
    order=[kind for kind in [*METHODS,'random','maintain'] if kind in scores.method.unique()]
    return per_run,per_run.groupby('method').mean().reindex(order)


def plot_highway_outcomes(study,save_dir=None):
    per_run,summary=study_summary(study);order=list(summary.index)
    fig,axes=plt.subplots(1,3,figsize=(14,5.2),layout='constrained',sharey=True)
    y=np.arange(len(order));colours=[COLOURS[kind] for kind in order]
    metrics=[('mean_reward','Held-out reward ↑','Total reward / episode',1),
             ('crash_rate','Held-out crashes ↓','Episodes ending in a crash (%)',100),
             ('duration_s','Episode duration ↑','Simulated seconds (maximum 30)',1)]
    for ax,(key,title,label,scale) in zip(axes,metrics):
        values=summary[key].to_numpy()*scale
        ax.barh(y,values,color=colours,height=.6)
        for row,kind in enumerate(order):
            points=per_run.loc[kind,key].to_numpy()*scale
            offsets=np.linspace(-.16,.16,len(points)) if len(points)>1 else np.zeros(1)
            ax.scatter(points,row+offsets,s=23,color='#302820',zorder=3)
            suffix='%' if scale==100 else ''
            ax.text(values[row]+.4,row-.32,f'{values[row]:.1f}{suffix}',fontsize=9)
        ax.set(title=title,xlabel=label);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.set_yticks(y,[method_name(kind,study) for kind in order]);ax.invert_yaxis() if ax is axes[0] else None
        maximum=max(values.max(),float(per_run[key].max())*scale)
        ax.set_xlim(0,114 if scale==100 else 32 if key=='duration_s' else max(maximum*1.22,1))
    n=len(study.metadata['test_seeds']);seeds=len(study.metadata['training_seeds'])
    fig.suptitle('Highway · final policies on new traffic',fontsize=15,weight='bold')
    fig.supxlabel(f'{study.metadata["steps_per_method"]:,} training interactions per learned policy · '
                  f'{seeds} independent training runs · {n} new traffic scenarios per policy\n'
                  'Dots = one mean per training run; vertical offsets separate overlaps. '
                  'Non-learning baselines use the same scenarios once.',fontsize=9)
    if save_dir is not None:fig.savefig(Path(save_dir)/'highway_comparison.png',dpi=180)
    plt.show()
    readable=pd.DataFrame({
        'Mean reward ↑':summary.mean_reward.round(2),
        'Crash rate ↓':summary.crash_rate.map(lambda x:f'{x:.1%}'),
        'Duration (s)':summary.duration_s.round(1),
        'CPU training (s/run)':summary.train_cpu_seconds.round(1),
        'Elapsed training (s/run)':summary.train_wall_seconds.round(1),
        'Stored table states':summary.table_states.map(lambda x:'N/A' if pd.isna(x) else f'{x:,.0f}'),
        'Test states found in table':summary.seen_fraction.map(lambda x:'N/A' if pd.isna(x) else f'{x:.1%}')
    }).rename(index={kind:method_name(kind,study) for kind in order})
    display(readable)
    return summary
