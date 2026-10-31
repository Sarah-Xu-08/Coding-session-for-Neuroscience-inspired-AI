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
from rl_lab.cafe import train_cafe,cafe_evaluate
from rl_lab.highway import highway_comparison,highway_evaluate,PlaybackDQN


def generate(reuse_training=False):
    torch.set_num_threads(1)
    definitions=load();rules=definitions['rules']
    folder=ROOT/'assets/reference_results';folder.mkdir(exist_ok=True)
    started=time.perf_counter()
    if not reuse_training:
        cafe,history=train_cafe(rules,episodes=60000,seed=0)
        history.to_csv(folder/'cafe_history_60000.csv',index=False)
        np.savez_compressed(ROOT/'assets/checkpoints/cafe_q_60000_seed0.npz',q=cafe.q,visits=cafe.visits)
        import pandas as pd
        scores={label:cafe_evaluate(policy,range(20000,20100)).mean()
                for label,policy in [('Tabular Q-learning',cafe),('Forecast-based rule',None)]}
        pd.DataFrame(scores).T.to_csv(folder/'cafe_scoreboard_60000.csv')
        agents,curves,scores=highway_comparison(rules,definitions['DQNAgent'],steps=2000,seeds=[0],eval_n=10)
        curves.to_csv(folder/'highway_learning_curves.csv',index=False)
        scores.to_csv(folder/'highway_heldout_evaluation.csv',index=False)
    else:
        for name in ['cafe_history_60000.csv','cafe_scoreboard_60000.csv',
                     'highway_learning_curves.csv','highway_heldout_evaluation.csv']:
            if not (folder/name).is_file():raise FileNotFoundError(folder/name)
    reference=highway_evaluate(PlaybackDQN(),range(92000,92020))
    reference.to_csv(folder/'reference_dqn_local_evaluation.csv',index=False)
    manifest={'platform':platform.platform(),'machine':platform.machine(),
        'python':platform.python_version(),'versions':{name:version(name) for name in
        ['numpy','torch','gymnasium','highway-env','pettingzoo','matplotlib','pandas']},
        'generated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'cafe':{'episodes':60000,'seed':0,'evaluation_seeds':[20000,20099]},
        'highway':{'steps_per_method':2000,'training_seeds':[0],
                   'development_seeds':[81000,81001,81002],'heldout_seeds':[91000,91009]},
        'reference_dqn':{'source':'supplied notebook checkpoint','training_steps':20000,
                         'training_seed':0,'local_evaluation_seeds':[92000,92019],
                         'local_mean_metrics':reference.mean().drop('seed').dropna().to_dict()},
        'generation_seconds':time.perf_counter()-started,
        'reused_training_files_for_checkpoint_reevaluation':reuse_training,
        'limitations':'One training seed. Short comparison is for teaching, not algorithm superiority.'}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    print('Recorded actual local runs and checkpoint re-evaluation.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-training',action='store_true',help='Keep existing training CSVs and re-evaluate only the reference checkpoint.')
    generate(parser.parse_args().reuse_training)
