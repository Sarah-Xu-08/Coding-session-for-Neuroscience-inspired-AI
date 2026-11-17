"""Record actual local measurements and re-evaluate the supplied checkpoint."""
import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import rl_lab
import numpy as np
import torch
from instructor_definitions import load
from rl_lab.cafe import CookieCafe,train_cafe,cafe_evaluate
from rl_lab.cafe_reference import solve_cafe_reference
from rl_lab.highway import highway_evaluate,PlaybackDQN


def generate_cafe_assets():
    """Refresh only the cookie assets, preserving recorded highway measurements."""
    import pandas as pd
    folder=ROOT/'assets/reference_results';folder.mkdir(exist_ok=True)
    cafe,history=train_cafe(load()['rules'],episodes=60000,seed=0)
    history.to_csv(folder/'cafe_history_60000.csv',index=False)
    reference=solve_cafe_reference()
    np.savez_compressed(ROOT/'assets/checkpoints/cafe_q_60000_seed0.npz',
        q=cafe.q,visits=cafe.visits,game=CookieCafe.GAME_ID,
        env_config=json.dumps(reference.env_kwargs))
    scores={label:cafe_evaluate(policy,range(20000,20100)).mean()
            for label,policy in [('Tabular Q-learning',cafe),('Forecast-based rule',None),
                                 ('Model-optimal reference',reference)]}
    pd.DataFrame(scores).T.to_csv(folder/'cafe_scoreboard_60000.csv')
    manifest_path=folder/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest['cafe']={'game':CookieCafe.GAME_ID,'product':CookieCafe.PRODUCT_NAME,
        'state':['hour','stock'],
        'q_shape':list(cafe.q.shape),'episodes':60000,'seed':0,'evaluation_seeds':[20000,20099],
        'demand_means':CookieCafe.MEANS.tolist(),
        'selling_hours':list(range(10,10+len(CookieCafe.MEANS))),
        'closing_hour':10+len(CookieCafe.MEANS),
        'only_uncertainty':'Poisson customer counts',
        'preparation_reliable':True,'env_config':reference.env_kwargs,
        'optimal_expected_return':reference.expected_opening_return,
        'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Recorded new cookie café assets; Q shape:',cafe.q.shape,flush=True)


def generate(reuse_training=False,workers=6):
    torch.set_num_threads(1)
    folder=ROOT/'assets/reference_results';folder.mkdir(exist_ok=True)
    started=time.perf_counter()
    if not reuse_training:
        from run_highway_study import run_pilot,run_full
        generate_cafe_assets()
        run_full(run_pilot(workers),workers)
    else:
        for name in ['cafe_history_60000.csv','cafe_scoreboard_60000.csv',
                     'highway_study/manifest.json']:
            if not (folder/name).is_file():raise FileNotFoundError(folder/name)
    reference=highway_evaluate(PlaybackDQN(),range(92000,92020))
    reference.to_csv(folder/'reference_dqn_local_evaluation.csv',index=False)
    manifest=json.loads((folder/'manifest.json').read_text())
    manifest.update(platform=platform.platform(),machine=platform.machine(),
        python=platform.python_version(),versions={name:version(name) for name in
        ['numpy','torch','gymnasium','highway-env','pettingzoo','matplotlib','pandas']},
        generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        reference_dqn={'source':'supplied notebook checkpoint','training_steps':20000,
                       'training_seed':0,'local_evaluation_seeds':[92000,92019],
                       'local_mean_metrics':reference.mean().drop('seed').dropna().to_dict()},
        generation_seconds=time.perf_counter()-started,
        reused_training_files_for_checkpoint_reevaluation=reuse_training)
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Recorded actual local runs and checkpoint re-evaluation.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-training',action='store_true',help='Keep existing training CSVs and re-evaluate only the reference checkpoint.')
    parser.add_argument('--workers',type=int,default=6,help='Parallel CPU workers for the highway pilot and study.')
    parser.add_argument('--cafe-only',action='store_true',help='Refresh cookie assets without retraining highway agents.')
    args=parser.parse_args()
    if args.cafe_only:generate_cafe_assets()
    else:generate(args.reuse_training,args.workers)
