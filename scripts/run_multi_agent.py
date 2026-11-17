"""Prepare genuine learned multi-agent policies outside the recording.

    .conda/bin/python scripts/run_multi_agent.py --algorithm all
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=['all','iql','mappo'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT/'assets/reference_results/multi_agent')
    parser.add_argument('--joint-steps', type=int, default=3000000)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true', help='Continue training from last.pt up to the total budget')
    parser.add_argument('--evaluate-test', action='store_true', help='Record final test only after development choices are fixed')
    parser.add_argument('--finalize-only', action='store_true', help='Evaluate the selected completed checkpoint without more training')
    args = parser.parse_args()
    if args.algorithm in ['all','iql']:
        from instructor_definitions import load
        from rl_lab.social_reference import save_social_reference
        *_, metadata = save_social_reference(load()['rules'], args.output, seed=args.seed)
        print('Independent Q-learning:', metadata['joint_decisions'], 'joint decisions', flush=True)
    if args.algorithm in ['all','mappo']:
        from rl_lab.mappo import train_mappo, finalize_mappo
        if args.finalize_only:
            finalize_mappo(args.output)
        else:
            train_mappo(args.output, joint_steps=args.joint_steps, num_envs=args.num_envs, seed=args.seed,
                        resume=args.resume, evaluate_test=args.evaluate_test)
