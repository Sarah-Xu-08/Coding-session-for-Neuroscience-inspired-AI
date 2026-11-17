"""Collect genuine SB3 results; retain the verified tabular runs on the same protocol."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import rl_lab
import pandas as pd
import torch
from rl_lab.library_dqn import (SB3_DQN_KWARGS, SB3_STUDY_DIR, train_library_dqn,
                               sb3_study_metadata)
from rl_lab.highway import highway_evaluate
from rl_lab.highway_study import (HighwayStudy, TABLE_FILES, load_cached_study,
                                 save_study, validate_study)
from run_highway_study import training_fingerprint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def run_seed(seed, folder, protocol):
    torch.set_num_threads(1)
    folder=Path(folder)/'runs'/f'DQN_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    completed=folder/'completed.json'
    if completed.exists():
        record=json.loads(completed.read_text())
        if record['protocol']==protocol and all(digest(folder/name)==value for name,value in record['files_sha256'].items()):
            return seed
    def checkpoint(step, recorder):
        write_json(folder/'progress.json',dict(step=step,seed=seed,complete=False))
    agent,curve=train_library_dqn(protocol['steps'],seed,protocol['eval_every'],protocol['eval_seeds'],
        verbose=True,on_checkpoint=checkpoint)
    reserved=set(protocol['eval_seeds'])|set(protocol['test_seeds'])|set(range(81000,81020))
    assert not reserved.intersection(agent.training_traffic_seeds)
    agent.training_episodes.to_csv(folder/'training_episodes.csv',index=False)
    agent.development_episodes.to_csv(folder/'development_episodes.csv',index=False)
    curve.assign(method='DQN',seed=seed).to_csv(folder/'evaluation_checkpoints.csv',index=False)
    highway_evaluate(agent,protocol['test_seeds']).assign(method='DQN',train_seed=seed,
        train_seconds=agent.training_seconds,train_cpu_seconds=agent.training_cpu_seconds,
        table_states=float('nan')).to_csv(folder/'test_episodes.csv',index=False)
    agent.save(folder/'agent.zip')
    write_json(folder/'traffic_seeds.json',agent.training_traffic_seeds)
    names=['training_episodes.csv','development_episodes.csv','evaluation_checkpoints.csv',
           'test_episodes.csv','agent.zip','traffic_seeds.json']
    write_json(completed,dict(protocol=protocol,files_sha256={name:digest(folder/name) for name in names}))
    write_json(folder/'progress.json',dict(step=protocol['steps'],seed=seed,complete=True))
    return seed


def collect(workers=5, folder=SB3_STUDY_DIR):
    old=load_cached_study()
    assert old.metadata['training_fingerprint']==training_fingerprint(), 'Verify tabular provenance before reuse.'
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    protocol=dict(steps=old.metadata['steps_per_method'],seeds=old.metadata['training_seeds'],
        eval_every=old.metadata['eval_every'],eval_seeds=old.metadata['development_seeds'],
        test_seeds=old.metadata['test_seeds'],dqn_kwargs=SB3_DQN_KWARGS,
        implementation_sha256=digest(ROOT/'rl_lab/library_dqn.py'),
        source_manifest_sha256=digest(ROOT/'assets/reference_results/highway_study/manifest.json'))
    write_json(folder/'protocol.json',protocol)
    print('Frozen SB3 protocol:',protocol['steps'],'steps; seeds',protocol['seeds'],flush=True)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(run_seed,seed,folder,protocol) for seed in protocol['seeds']]
        for future in as_completed(futures):
            print('Completed SB3 seed',future.result(),flush=True)
    tables={}
    for key,name in TABLE_FILES.items():
        original=getattr(old,key)
        if key=='baselines':tables[key]=original.copy();continue
        parts=[original.loc[original.method!='DQN'].copy()]
        parts.extend(pd.read_csv(folder/'runs'/f'DQN_seed{seed}'/name) for seed in protocol['seeds'])
        tables[key]=pd.concat(parts,ignore_index=True)
    metadata=sb3_study_metadata(protocol['steps'],protocol['seeds'],protocol['eval_every'],
        protocol['eval_seeds'],protocol['test_seeds'],SB3_DQN_KWARGS,old.metadata['table_kwargs'])
    metadata.update(source='Measured SB3 runs with scripts/run_sb3_study.py; verified tabular/baseline records reused.',
        reused_tabular_source='assets/reference_results/highway_study',
        reused_tabular_manifest_sha256=protocol['source_manifest_sha256'],
        tabular_settings_selection='Earlier development-only tabular pilot; no equal SB3 tuning claim.',
        training_traffic_disjoint_from_reserved_seeds=True,
        completed_utc=datetime.now(timezone.utc).isoformat(),workers=workers,
        implementation_sha256=protocol['implementation_sha256'])
    study=validate_study(HighwayStudy(**tables,metadata=metadata))
    # Include all final policies, while preserving the original study unchanged.
    for kind in ['coarse','medium','fine']:
        for seed in protocol['seeds']:
            destination=folder/'runs'/f'{kind}_seed{seed}';destination.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/'assets/reference_results/highway_study/runs'/f'{kind}_seed{seed}'/'agent.npz',destination/'agent.npz')
    save_study(study,folder)
    manifest=json.loads((folder/'manifest.json').read_text())
    model_files=list(folder.glob('runs/*/agent.*'))
    manifest['model_files_sha256']={str(p.relative_to(folder)):digest(p) for p in model_files}
    manifest['files_sha256'].update(manifest['model_files_sha256'])
    write_json(folder/'manifest.json',manifest)
    print('Published verified SB3 study:',folder,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=5)
    args=parser.parse_args()
    collect(args.workers)
