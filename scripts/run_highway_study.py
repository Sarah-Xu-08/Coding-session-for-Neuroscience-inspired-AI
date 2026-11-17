"""Reproducible offline pilot and repeated-seed study; resume completed runs.

Example: python scripts/run_highway_study.py --phase all --workers 6
An interrupted policy restarts; finished, fingerprint-matching policies are reused.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
from importlib.metadata import version
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import shutil
import sys
import time

# Set before importing NumPy/PyTorch, including in spawned workers.
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from instructor_definitions import load
from rl_lab.highway import HIGHWAY_CONFIG, SparseTrafficQ, train_highway, highway_evaluate
from rl_lab.highway_study import (HighwayStudy, METHODS, TABLE_FILES, baseline_evaluation,
                                  save_study, validate_study)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def training_fingerprint():
    """Hash training definitions, not plots, notebook outputs or prose."""
    import nbformat
    wanted={'epsilon_greedy','td_target','q_update','dqn_targets','dqn_loss',
            'QNetwork','ReplayBuffer','DQNAgent','random_argmax','TrafficFeatures',
            'make_highway','TrafficDiscretizer','SparseTrafficQ','highway_evaluate','train_highway'}
    nodes=[]
    notebook=nbformat.read(ROOT/'notebooks/chapter3_instructor.ipynb',as_version=4)
    sources=[c.source for c in notebook.cells if c.cell_type=='code']
    sources.append((ROOT/'rl_lab/highway.py').read_text())
    for source in sources:
        for node in ast.parse(source).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in wanted:
                nodes.append(ast.dump(node,include_attributes=False))
    payload=dict(definitions=nodes,config=HIGHWAY_CONFIG,runner_protocol=1,
                 versions={name:version(name) for name in ['numpy','torch','gymnasium','highway-env']})
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def specification(phase, kind, seed, kwargs, fingerprint, label=None):
    pilot=phase=='pilot'
    return dict(phase=phase,method=kind,seed=seed,label=label or kind,
        steps=10000 if pilot else 50000,eval_every=2000 if pilot else 1000,
        development_seeds=list(range(81000,81020)) if pilot else list(range(82000,82020)),
        test_seeds=[] if pilot else list(range(93000,93100)),
        agent_kwargs=kwargs,training_fingerprint=fingerprint)


def run_name(spec):
    return f'{spec["label"]}_seed{spec["seed"]}'


def completed_run(folder, spec):
    path=Path(folder)/'completed.json'
    if not path.exists():return False
    metadata=json.loads(path.read_text())
    if metadata.get('specification')!=spec:return False
    hashes=metadata.get('files_sha256',{})
    return bool(hashes) and all((Path(folder)/name).is_file() and digest(Path(folder)/name)==expected
                                for name,expected in hashes.items())


def worker(spec, folder):
    torch.set_num_threads(1)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if training_fingerprint()!=spec['training_fingerprint']:
        raise RuntimeError('Training definitions changed after the study was launched.')
    definitions=load();started=time.perf_counter()
    def checkpoint(agent,curve,episodes,development):
        # Journal actual observations for monitoring; final files are written below.
        curve.to_csv(folder/'evaluation_checkpoints.csv',index=False)
        episodes.to_csv(folder/'training_episodes.csv',index=False)
        development.to_csv(folder/'development_episodes.csv',index=False)
        last=curve.iloc[-1]
        atomic_json(folder/'progress.json',dict(step=int(last.step),
            development_reward=float(last.return_),development_crash_rate=float(last.collision),
            elapsed_seconds=time.perf_counter()-started,complete=False))
    agent,curve=train_highway(spec['method'],definitions['rules'],definitions['DQNAgent'],
        steps=spec['steps'],seed=spec['seed'],eval_every=spec['eval_every'],
        eval_seeds=spec['development_seeds'],agent_kwargs=spec['agent_kwargs'],
        on_checkpoint=checkpoint,verbose=False)
    curve.assign(method=spec['method'],seed=spec['seed']).to_csv(folder/'evaluation_checkpoints.csv',index=False)
    agent.training_episodes.to_csv(folder/'training_episodes.csv',index=False)
    agent.development_episodes.to_csv(folder/'development_episodes.csv',index=False)
    if spec['test_seeds']:
        scores=highway_evaluate(agent,spec['test_seeds']).assign(
            method=spec['method'],train_seed=spec['seed'],train_seconds=agent.training_seconds,
            train_cpu_seconds=agent.training_cpu_seconds,
            table_states=len(agent.q) if isinstance(agent,SparseTrafficQ) else np.nan)
        scores.to_csv(folder/'test_episodes.csv',index=False)
    target=folder/'agent.npz'
    if spec['method']=='DQN':
        np.savez_compressed(target,**{k:v.detach().cpu().numpy() for k,v in agent.online.state_dict().items()})
    else:
        np.savez_compressed(target,keys=np.asarray(list(agent.q),dtype=np.int32),
                            q=np.asarray(list(agent.q.values()),dtype=np.float32))
    atomic_json(target.with_suffix('.json'),dict(config=HIGHWAY_CONFIG,sha256=digest(target),
        training_steps=spec['steps'],training_seed=spec['seed'],method=spec['method'],
        agent_kwargs=spec['agent_kwargs'],training_fingerprint=spec['training_fingerprint']))
    files=['training_episodes.csv','development_episodes.csv','evaluation_checkpoints.csv','agent.npz','agent.json']
    if spec['test_seeds']:files.append('test_episodes.csv')
    atomic_json(folder/'completed.json',dict(specification=spec,
        training_seconds=agent.training_seconds,training_cpu_seconds=agent.training_cpu_seconds,
        elapsed_seconds=time.perf_counter()-started,files_sha256={name:digest(folder/name) for name in files}))
    atomic_json(folder/'progress.json',dict(step=spec['steps'],complete=True,
                                           elapsed_seconds=time.perf_counter()-started))
    return run_name(spec)


def execute(specs, folder, workers):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    pending=[spec for spec in specs if not completed_run(folder/run_name(spec),spec)]
    print(f'{len(specs)-len(pending)}/{len(specs)} matching policies already finished; '
          f'{len(pending)} to run with {workers} workers.',flush=True)
    if not pending:return
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(worker,spec,str(folder/run_name(spec))):spec for spec in pending}
        reported=time.perf_counter()
        while futures:
            done,_=wait(futures,timeout=10,return_when=FIRST_COMPLETED)
            for future in done:
                spec=futures.pop(future)
                print('Finished:',future.result(),flush=True)
            if time.perf_counter()-reported>=30 or not futures:
                steps=0;finished=0
                for spec in specs:
                    progress=folder/run_name(spec)/'progress.json'
                    if progress.exists():
                        data=json.loads(progress.read_text());steps+=data['step'];finished+=int(data['complete'])
                print(f'Progress: {finished}/{len(specs)} policies; {steps:,}/'
                      f'{sum(spec["steps"] for spec in specs):,} training interactions.',flush=True)
                reported=time.perf_counter()


def run_pilot(workers):
    folder=ROOT/'results/highway_study/pilot';fingerprint=training_fingerprint()
    candidates=[('DQN','DQN_lr_0.001',{'learning_rate':.001}),
                ('DQN','DQN_lr_0.0003',{'learning_rate':.0003}),
                ('medium','medium_alpha_0.15',{'alpha':.15}),
                ('medium','medium_alpha_0.3',{'alpha':.3})]
    specs=[specification('pilot',kind,seed,kwargs,fingerprint,label)
           for kind,label,kwargs in candidates for seed in [110,111]]
    execute(specs,folder,workers)
    rows=[]
    for kind,label,kwargs in candidates:
        curves=pd.concat([pd.read_csv(folder/f'{label}_seed{seed}'/'evaluation_checkpoints.csv')
                          for seed in [110,111]])
        last=curves[curves.step>=6000]
        rows.append(dict(method=kind,label=label,agent_kwargs=kwargs,
            development_reward=float(last.return_.mean()),development_crash_rate=float(last.collision.mean())))
    selected={}
    for kind in ['DQN','medium']:
        ranked=sorted([row for row in rows if row['method']==kind],
                      key=lambda row:(-row['development_reward'],row['development_crash_rate'],row['label']))
        selected[kind]=ranked[0]['agent_kwargs']
    result=dict(training_fingerprint=fingerprint,steps_per_policy=10000,training_seeds=[110,111],
        development_seeds=list(range(81000,81020)),test_seeds_used=[],
        selection_rule='Highest mean development reward over the last three checkpoints and both training seeds; crash rate breaks ties.',
        candidates=rows,dqn_kwargs=selected['DQN'],table_kwargs=selected['medium'],
        table_note='Medium bins tuned; the selected alpha is also used for the coarse/fine sensitivity checks.')
    atomic_json(folder/'selection.json',result)
    print('Pilot selection:',json.dumps(selected),flush=True)
    return result


def run_full(selection, workers):
    started=time.perf_counter();fingerprint=training_fingerprint()
    if selection['training_fingerprint']!=fingerprint:raise ValueError('Repeat pilot: training definitions changed.')
    folder=ROOT/'results/highway_study/full'
    specs=[specification('full',kind,seed,selection['dqn_kwargs'] if kind=='DQN' else selection['table_kwargs'],fingerprint)
           for seed in range(5) for kind in METHODS]
    # Freeze before opening held-out results, even when resuming a completed run.
    atomic_json(folder/'protocol.json',dict(specifications=specs,pilot=selection))
    execute(specs,folder,workers)
    parts={key:[] for key in ['training','evaluation','curves','scores']}
    for spec in specs:
        source=folder/run_name(spec)
        for key in parts:parts[key].append(pd.read_csv(source/TABLE_FILES[key]))
    metadata=dict(experiment_id='highway_episode_study_v1',complete=True,
        steps_per_method=50000,training_seeds=list(range(5)),development_seeds=list(range(82000,82020)),
        test_seeds=list(range(93000,93100)),eval_every=1000,methods=list(METHODS),
        dqn_kwargs=selection['dqn_kwargs'],table_kwargs=selection['table_kwargs'],pilot=selection,
        source='Measured locally with scripts/run_highway_study.py',config=HIGHWAY_CONFIG,
        training_fingerprint=fingerprint,workers=workers,device='cpu',
        platform=platform.platform(),machine=platform.machine(),python=platform.python_version(),
        versions={name:version(name) for name in ['numpy','torch','gymnasium','highway-env','pandas']},
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        collection_elapsed_seconds=time.perf_counter()-started,
        limitations='One fixed environment; five training seeds. SD bands describe run variation, not confidence intervals. Concurrent elapsed training times depend on system load.')
    baseline_path=folder/'baseline_episodes.csv'
    if baseline_path.exists():baselines=pd.read_csv(baseline_path)
    else:
        baselines=baseline_evaluation(metadata['development_seeds'],metadata['test_seeds'])
        baselines.to_csv(baseline_path,index=False)
    study=HighwayStudy(**{key:pd.concat(frames,ignore_index=True) for key,frames in parts.items()},
                       baselines=baselines,metadata=metadata)
    validate_study(study)
    # Reconstruct the deterministic reset stream and audit that no actual training
    # traffic seed equals a reserved development or test seed.
    reserved=set(metadata['development_seeds']+metadata['test_seeds'])
    for (method,seed),episodes in study.training.groupby(['method','train_seed']):
        reset_seeds=np.random.default_rng(int(seed)+50000).integers(2**31,size=len(episodes))
        if reserved.intersection(map(int,reset_seeds)):
            raise RuntimeError(f'Training traffic overlapped a reserved seed: {method}, {seed}')
    metadata['training_traffic_disjoint_from_reserved_seeds']=True
    metadata['total_training_cpu_seconds']=float(study.curves.groupby(['method','seed']).train_cpu_seconds.max().sum())
    metadata['sum_individual_run_elapsed_seconds']=sum(json.loads((folder/run_name(spec)/'completed.json').read_text())['elapsed_seconds'] for spec in specs)
    target=ROOT/'assets/reference_results/highway_study'
    metadata['complete']=False
    save_study(study,target)
    for spec in specs:
        shutil.copytree(folder/run_name(spec),target/'runs'/run_name(spec),dirs_exist_ok=True)
    shutil.copytree(ROOT/'results/highway_study/pilot',target/'pilot',dirs_exist_ok=True)
    shutil.copy2(folder/'protocol.json',target/'protocol.json')
    metadata['collection_elapsed_seconds']=time.perf_counter()-started
    metadata['complete']=True
    metadata['files_sha256']={str(path.relative_to(target)):digest(path) for path in target.rglob('*')
                              if path.is_file() and path.name!='manifest.json'}
    atomic_json(target/'manifest.json',metadata)
    parent=target.parent/'manifest.json'
    manifest=json.loads(parent.read_text())
    manifest['legacy_highway']=manifest.get('legacy_highway',manifest.get('highway',{}))
    manifest['highway']={key:metadata[key] for key in ['experiment_id','steps_per_method','training_seeds','development_seeds','test_seeds','eval_every','dqn_kwargs','table_kwargs']}
    manifest['highway']['folder']='highway_study'
    manifest['limitations']=metadata['limitations']
    atomic_json(parent,manifest)
    print('Saved complete study:',target,flush=True)
    return study


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['pilot','full','all'],default='all')
    parser.add_argument('--workers',type=int,default=6)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    if args.phase in ['pilot','all']:selection=run_pilot(args.workers)
    else:selection=json.loads((ROOT/'results/highway_study/pilot/selection.json').read_text())
    if args.phase in ['full','all']:run_full(selection,args.workers)
