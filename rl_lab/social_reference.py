"""Reproducible Independent Q-learning example, using the lesson's Q update."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT
from .social import SocialQ, social_match, train_social

REFERENCE = ROOT / 'assets/reference_results/multi_agent'


def save_social_reference(rules, folder=REFERENCE, episodes=3000, seed=0):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    a, b, history = train_social('prisoners_dilemma', rules, episodes=episodes, seed=seed)
    data = {name: [dict(state=list(state), values=values.tolist()) for state, values in agent.q.items()]
            for name, agent in [('player_a', a), ('player_b', b)]}
    (folder/'independent_q.json').write_text(json.dumps(data, indent=2))
    history.to_csv(folder/'independent_q_training.csv', index=False)
    evaluation_rng = [agent.rng.bit_generator.state for agent in (a, b)]
    evaluation = social_match('prisoners_dilemma', a, b, nrounds=20, learn=False)
    evaluation.to_csv(folder/'independent_q_selfplay.csv', index=False)
    names = ['independent_q.json', 'independent_q_training.csv', 'independent_q_selfplay.csv']
    metadata = dict(algorithm='Independent Q-learning', training_seed=seed,
                    episodes=episodes, rounds=20, joint_decisions=episodes*20,
                    state=['rounds_remaining','own_previous_action','partner_previous_action'],
                    alpha=.15, gamma=.95, epsilon_start=.5, epsilon_end=.03,
                    training='two separate Q-tables, each updated from its own payoff; simultaneous choices',
                    evaluation='frozen Q-tables; epsilon=0; all matchups use 20 rounds',
                    selection='final checkpoint; no model selection',
                    evaluation_rng_states=evaluation_rng,
                    files_sha256={n: hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in names})
    (folder/'independent_q_manifest.json').write_text(json.dumps(metadata, indent=2))
    return a, b, history, metadata


def load_social_reference(rules, folder=REFERENCE):
    folder = Path(folder)
    metadata = json.loads((folder/'independent_q_manifest.json').read_text())
    for name, digest in metadata['files_sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Independent Q reference file changed: {name}')
    data = json.loads((folder/'independent_q.json').read_text())
    agents = []
    for i, name in enumerate(['player_a','player_b']):
        agent = SocialQ(rules, seed=1000+i)
        agent.q = {tuple(row['state']): np.asarray(row['values'], dtype=float) for row in data[name]}
        agent.rng.bit_generator.state = metadata['evaluation_rng_states'][i]
        agents.append(agent)
    return *agents, pd.read_csv(folder/'independent_q_training.csv'), metadata
