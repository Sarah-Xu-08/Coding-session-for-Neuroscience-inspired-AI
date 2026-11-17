"""Provided HighwayEnv preprocessing, sparse tables, experiments and playback."""
import copy
import io
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import highway_env
from highway_env.envs.common.graphics import EnvViewer
import torch
from torch import nn
from PIL import Image as PILImage
from IPython.display import display, Image
from . import ROOT

AGENT_BROWN = '#9a5b2f'
AGENT_RGB = (154, 91, 47)
TRAFFIC_RGB = (170, 170, 170)

def random_argmax(values, rng):
    values=np.asarray(values)
    return int(rng.choice(np.flatnonzero(values == values.max())))
#@title Existing HighwayEnv, shared observations, and sparse tabular baselines
HIGHWAY_CONFIG = {
    'lanes_count':3, 'vehicles_count':18, 'duration':30,
    'observation':{'type':'Kinematics','vehicles_count':5,
                   'features':['presence','x','y','vx','vy'],
                   'absolute':False,'normalize':False,'clip':False,'order':'sorted'},
    'action':{'type':'DiscreteMetaAction'},
    'simulation_frequency':5, 'policy_frequency':1,
}

ACTION_NAMES = ['Lane left','Maintain','Lane right','Faster','Slower']

class TrafficFeatures(gym.ObservationWrapper):
    """Shared preprocessing for ALL highway agents; no privileged simulator state.

    World x of the ego vehicle is set to zero (translation invariance). All other
    observed vehicle features are retained. y is in lane-width units; vx and vy
    are scaled by 30 and 10 m/s. Clipping is identical for both learners.
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(-3.,3.,shape=(25,),dtype=np.float32)

    def observation(self, observation):
        z=np.asarray(observation,dtype=np.float32).copy()
        z[0,1]=0.
        z[:,1]=np.clip(z[:,1]/100.,-2.,2.)
        z[:,2]=np.clip(z[:,2]/4.,-3.,3.)
        z[:,3]=np.clip(z[:,3]/30.,-2.,2.)
        z[:,4]=np.clip(z[:,4]/10.,-2.,2.)
        z[z[:,0]<0.5]=0.
        return z.reshape(-1)

    def render(self):
        """Draw actual simulator pixels offscreen, including with SDL's dummy driver.

        HighwayEnv 1.10 disables its viewer under SDL_VIDEODRIVER=dummy even
        for RGB output. Enable drawing on its offscreen surface explicitly;
        no native window or change to the simulation is needed.
        """
        if self.render_mode == 'rgb_array':
            env = self.unwrapped
            if env.viewer is None:
                env.viewer = EnvViewer(env)
            env.viewer.enabled = True
            controlled = {id(vehicle) for vehicle in env.controlled_vehicles}
            for vehicle in env.road.vehicles:
                vehicle.color = AGENT_RGB if id(vehicle) in controlled else TRAFFIC_RGB
        frame = self.env.render()
        if self.render_mode == 'rgb_array' and (frame is None or not np.any(frame)):
            raise RuntimeError('Highway rendering produced a blank frame. Restart the kernel and rerun setup.')
        return frame

def make_highway(render=False):
    config = copy.deepcopy(HIGHWAY_CONFIG)
    if render:
        # Visual option only: leave HIGHWAY_CONFIG/checkpoint dynamics unchanged.
        config['offscreen_rendering'] = True
    env=gym.make('highway-fast-v0',config=config,
                 render_mode='rgb_array' if render else None)
    return TrafficFeatures(env)

class TrafficDiscretizer:
    """A sparse table key with feature-aware, symmetric physical bins.

    Preserve presence. Preserve y at quarter-lane precision in every resolution
    so that coarse bins do not erase which lane a vehicle occupies.
    """
    SPECS={
        'coarse':(np.array([0,15,40,100,200]),5.,2.),
        'medium':(np.array([0,5,15,30,60,100,200]),2.5,1.),
        'fine':(np.array([0,2,5,10,15,25,40,60,100,150,200]),1.,0.5),
    }
    def __init__(self,resolution='medium'):
        self.resolution=resolution
        self.distance,self.speed_step,self.lateral_step=self.SPECS[resolution]
        self.edges=np.unique(np.r_[-self.distance[::-1],self.distance])

    def __call__(self,observation):
        key=[]
        for presence,x,y,vx,vy in np.asarray(observation).reshape(5,5):
            if presence<.5:
                key.extend((0,0,0,0,0));continue
            key.extend((1,int(np.digitize(x*100.,self.edges)),
                        int(np.rint(y*4.)),
                        int(np.rint(vx*30./self.speed_step)),
                        int(np.rint(vy*10./self.lateral_step))))
        return tuple(key)

class SparseTrafficQ:
    def __init__(self,rules,resolution='medium',seed=0,alpha=.15,gamma=.95):
        self.rules=rules
        self.encoder=TrafficDiscretizer(resolution)
        self.q={};self.rng=np.random.default_rng(seed)
        self.alpha,self.gamma=alpha,gamma

    def values(self,observation,create=False):
        key=self.encoder(observation)
        if create and key not in self.q:self.q[key]=np.zeros(5,np.float32)
        return self.q.get(key,np.zeros(5,np.float32))

    def act(self,observation,epsilon=0.):
        return self.rules.epsilon_greedy(self.values(observation),epsilon,self.rng)

    def update(self,state,action,reward,next_state,terminated):
        row=self.values(state,create=True)
        target=self.rules.td_target(reward,terminated,self.values(next_state),np.ones(5,bool),self.gamma)
        row[action]=self.rules.q_update(row[action],target,self.alpha)

def highway_evaluate(agent,seeds):
    env=make_highway()
    rows=[]
    rng_state=copy.deepcopy(agent.rng.bit_generator.state) if hasattr(agent,'rng') else None
    try:
        for seed in seeds:
            if rng_state is not None:
                agent.rng.bit_generator.state=np.random.default_rng(int(seed)+800000).bit_generator.state
            state,info=env.reset(seed=int(seed))
            total=0.;speeds=[];seen=0;steps=0;changes=0
            done=truncated=False
            while not(done or truncated):
                if isinstance(agent,SparseTrafficQ):seen+=int(agent.encoder(state) in agent.q)
                action=agent.act(state) if hasattr(agent,'act') else int(agent(state))
                state,reward,done,truncated,info=env.step(action)
                total+=reward;steps+=1
                speeds.append(float(info.get('speed',0.)))
                changes+=int(action in [0,2])
            rows.append(dict(seed=int(seed),return_=total,collision=int(info.get('crashed',False)),
                             mean_speed_m_s=float(np.mean(speeds)),steps=steps,
                             duration_s=steps/HIGHWAY_CONFIG['policy_frequency'],
                             seen_fraction=seen/max(steps,1) if isinstance(agent,SparseTrafficQ) else np.nan,
                             lane_change_requests=changes))
    finally:
        env.close()
        if rng_state is not None:agent.rng.bit_generator.state=rng_state
    return pd.DataFrame(rows)

def train_highway(kind,rules,dqn_factory=None,steps=2000,seed=0,eval_every=None,eval_seeds=(81000,81001,81002),
                  device='cpu',verbose=True,agent_kwargs=None,on_checkpoint=None):
    """Train with episode-level returns and separate, unbinned greedy evaluations.

    The returned agent holds training_episodes and development_episodes tables.
    A final partial episode is saved for accounting, but excluded from the EMA.
    """
    if steps<1 or (eval_every is not None and eval_every<1):
        raise ValueError('Training steps and evaluation interval must be positive.')
    eval_seeds=tuple(map(int,eval_seeds))
    if not eval_seeds:raise ValueError('Use at least one development traffic seed.')
    env=make_highway()
    if kind == 'DQN' and dqn_factory is None:
        raise ValueError('Supply the DQNAgent class defined in your notebook.')
    config=dict(agent_kwargs or {})
    agent=dqn_factory(seed=seed,device=device,**config) if kind=='DQN' else SparseTrafficQ(rules,kind,seed=seed,**config)
    eval_every=eval_every or max(steps//4,1)
    resets=np.random.default_rng(seed+50000)
    state,_=env.reset(seed=int(resets.integers(2**31)))
    logs=[];episodes=[];development=[]
    started=time.perf_counter();cpu_started=time.process_time();paused=cpu_paused=0.
    episode_return=0.;episode_length=0;episode_seen=0;episode_first_epsilon=1.
    checkpoints=sorted(set([0,*range(eval_every,steps+1,eval_every),steps]))
    def record(step):
        nonlocal paused,cpu_paused
        evaluation_started=time.perf_counter();evaluation_cpu=time.process_time()
        train_wall=evaluation_started-started-paused
        train_cpu=evaluation_cpu-cpu_started-cpu_paused
        frame=highway_evaluate(agent,eval_seeds)
        development.extend(frame.assign(step=step,train_seed=seed,method=kind).to_dict('records'))
        logs.append(dict(step=step,return_=frame.return_.mean(),collision=frame.collision.mean(),
                         train_seconds=train_wall,train_cpu_seconds=train_cpu,
                         eval_episodes=len(frame),mean_duration_s=frame.duration_s.mean(),
                         seen_fraction=frame.seen_fraction.mean()))
        if on_checkpoint is not None:
            on_checkpoint(agent,pd.DataFrame(logs),pd.DataFrame(episodes),pd.DataFrame(development))
        paused+=time.perf_counter()-evaluation_started
        cpu_paused+=time.process_time()-evaluation_cpu
        if verbose: print(f'{kind:>6} seed={seed} step={step:>6,} '
                          f'eval return={frame.return_.mean():5.1f} '
                          f'collisions={frame.collision.mean():.0%}',flush=True)
    record(0)
    for step in range(1,steps+1):
        epsilon=float(np.interp(step,[0,steps*.7],[1.,.05]))
        if episode_length==0:episode_first_epsilon=epsilon
        if isinstance(agent,SparseTrafficQ):episode_seen+=int(agent.encoder(state) in agent.q)
        action=agent.act(state,epsilon)
        nxt,reward,terminated,truncated,info=env.step(action)
        episode_return+=reward;episode_length+=1
        if kind=='DQN':
            agent.buffer.add(state,action,reward,nxt,terminated)
            if step>=256 and step%4==0:agent.learn()
            if step%400==0:agent.sync_target()
        else:
            agent.update(state,action,reward,nxt,terminated)
        state=nxt
        if terminated or truncated:
            episodes.append(dict(episode=len(episodes)+1,step=step,return_=episode_return,
                                 length=episode_length,collision=int(info.get('crashed',False)),
                                 complete=True,epsilon_start=episode_first_epsilon,epsilon_end=epsilon,
                                 seen_fraction=episode_seen/episode_length if isinstance(agent,SparseTrafficQ) else np.nan,
                                 train_seed=seed,method=kind))
            episode_return=0.;episode_length=episode_seen=0
            if step<steps:state,_=env.reset(seed=int(resets.integers(2**31)))
        elif step==steps:
            episodes.append(dict(episode=len(episodes)+1,step=step,return_=episode_return,
                                 length=episode_length,collision=int(info.get('crashed',False)),
                                 complete=False,epsilon_start=episode_first_epsilon,epsilon_end=epsilon,
                                 seen_fraction=episode_seen/episode_length if isinstance(agent,SparseTrafficQ) else np.nan,
                                 train_seed=seed,method=kind))
        if step in checkpoints:record(step)
    env.close()
    agent.training_steps=steps
    agent.training_seconds=time.perf_counter()-started-paused
    agent.training_cpu_seconds=time.process_time()-cpu_started-cpu_paused
    agent.training_episodes=pd.DataFrame(episodes)
    agent.development_episodes=pd.DataFrame(development)
    agent.training_config=dict(steps=steps,seed=seed,eval_every=eval_every,
                               eval_seeds=list(eval_seeds),agent_kwargs=config)
    return agent,pd.DataFrame(logs)

def highway_comparison(rules,dqn_factory,steps=2000,seeds=(0,),eval_n=5,device='cpu'):
    agents={};curves=[];scores=[]
    for seed in seeds:
        for kind in ['coarse','medium','fine','DQN']:
            agent,curve=train_highway(kind,rules,dqn_factory,steps=steps,seed=seed,device=device)
            curve['method']=kind;curve['seed']=seed;curves.append(curve)
            test=highway_evaluate(agent,range(91000,91000+eval_n))
            test['method']=kind;test['train_seed']=seed
            test['train_seconds']=agent.training_seconds
            test['table_states']=len(agent.q) if isinstance(agent,SparseTrafficQ) else np.nan
            scores.append(test);agents[(kind,seed)]=agent
    return agents,pd.concat(curves,ignore_index=True),pd.concat(scores,ignore_index=True)

def plot_highway_comparison(curves,scores,save_dir=None):
    """Lead with medium-bin Q-learning versus DQN; show every final baseline."""
    order = [kind for kind in ['coarse', 'medium', 'fine', 'DQN'] if kind in set(scores.method)]
    names = {'coarse': 'Tabular · coarse bins', 'medium': 'Tabular · medium bins',
             'fine': 'Tabular · fine bins', 'DQN': 'DQN · neural network'}
    palette = {'coarse': '#c7c7c7', 'medium': '#626262', 'fine': '#a0a0a0', 'DQN': AGENT_BROWN}
    summary=scores.groupby('method',sort=False).agg(
        mean_return=('return_','mean'),collision_rate=('collision','mean'),
        mean_speed_m_s=('mean_speed_m_s','mean'),training_seconds=('train_seconds','mean'),
        table_states=('table_states','mean'),eval_state_coverage=('seen_fraction','mean')).reindex(order)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    learning, reward_ax, crash_ax, time_ax = axes.flat
    # Medium bins are fixed in advance for the main comparison, not chosen by score.
    primary = [kind for kind in ['medium', 'DQN'] if kind in set(curves.method)]
    if not primary:
        primary = order
    for kind in primary:
        frame = curves[curves.method == kind]
        agg=frame.groupby('step')['return_'].agg(['mean','min','max'])
        learning.plot(agg.index, agg['mean'], label=names[kind], color=palette[kind],
                      marker='o' if kind == 'DQN' else 's',
                      linestyle='-' if kind == 'DQN' else '--', lw=2.4)
        if frame.seed.nunique()>1:
            learning.fill_between(agg.index, agg['min'], agg['max'], alpha=.15, color=palette[kind])
    learning.set(title='1 · Learning on development traffic ↑',
                 xlabel='Training interactions', ylabel='Mean evaluation reward / episode')
    learning.legend(frameon=False, fontsize=9)
    learning.grid(axis='y', alpha=.15)

    y = np.arange(len(order))
    colours = [palette[kind] for kind in order]
    for ax in [reward_ax, crash_ax, time_ax]:
        ax.set_yticks(y, [names[kind] for kind in order], fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.15)
        ax.set_axisbelow(True)
    means = summary.mean_return.to_numpy()
    reward_ax.barh(y, means, color=colours, height=.52, alpha=.65)
    for row, kind in enumerate(order):
        values = scores.loc[scores.method == kind, 'return_'].to_numpy()
        offsets = np.linspace(-.16, .16, len(values)) if len(values) > 1 else [0.]
        reward_ax.scatter(values, row + np.asarray(offsets), s=18,
                          color=AGENT_BROWN if kind == 'DQN' else '#454545', alpha=.8, zorder=3)
        reward_ax.text(means[row] + .25, row - .30, f'mean {means[row]:.1f}',
                       fontsize=9, color=palette[kind] if kind == 'DQN' else '#333333')
    reward_ax.set(title='2 · Reward on NEW traffic ↑', xlabel='Total reward / episode')
    reward_ax.set_xlim(left=0, right=max(scores.return_.max(), means.max()) * 1.15)

    crash_rates = summary.collision_rate.to_numpy() * 100
    crash_ax.barh(y, crash_rates, color=colours, height=.52)
    for row, value in enumerate(crash_rates):
        crash_ax.text(value + 1.5, row, f'{value:.0f}%', va='center', fontsize=10)
    crash_ax.set(title='3 · Crashes on the same NEW traffic ↓', xlabel='Episodes ending in a crash (%)',
                 xlim=(0, 114), xticks=[0, 25, 50, 75, 100])

    times = summary.training_seconds.to_numpy()
    time_ax.barh(y, times, color=colours, height=.52)
    for row, value in enumerate(times):
        time_ax.text(value + max(times.max() * .02, .01), row, f'{value:.1f}s', va='center', fontsize=10)
    time_ax.set(title='4 · Training time on this CPU ↓', xlabel='Seconds (evaluation time excluded)',
                xlim=(0, max(times.max() * 1.22, .1)))
    fig.suptitle('Highway · tabular Q-learning versus DQN', fontsize=16, weight='bold')
    counts = scores.groupby(['method', 'train_seed']).size()
    episode_count = str(counts.iloc[0]) if counts.nunique() == 1 else 'multiple'
    seed_count = curves.seed.nunique()
    seed_label = 'training seed' if seed_count == 1 else 'training seeds'
    fig.supxlabel(
        f'{int(curves.step.max()):,} interactions per method · {seed_count} {seed_label} · '
        f'{episode_count} held-out traffic episodes per trained agent\n'
        'Dots = individual test episodes. Short runs can perform poorly; compare crashes alongside reward.',
        fontsize=10)
    if save_dir is not None:
        fig.savefig(Path(save_dir) / 'highway_comparison.png', dpi=180)
    plt.show()
    readable = pd.DataFrame({
        'Mean reward ↑': summary.mean_return.round(2),
        'Crash rate ↓': summary.collision_rate.map(lambda value: f'{value:.0%}'),
        'Mean speed (m/s)': summary.mean_speed_m_s.round(1),
        'Training time (s)': summary.training_seconds.round(1),
        'Stored table states': summary.table_states.map(lambda value: 'N/A' if pd.isna(value) else f'{int(value):,}'),
        'Test states found in table': summary.eval_state_coverage.map(lambda value: 'N/A' if pd.isna(value) else f'{value:.1%}'),
    }).rename(index=names)
    display(readable)
    return summary

def highway_rollout(agent,seed=2026,max_frames=100,save_path=None):
    env=make_highway(render=True)
    state,info=env.reset(seed=seed)
    frames=[PILImage.fromarray(env.render())]
    records=[]
    done=truncated=False
    while not(done or truncated):
        action=agent.act(state) if hasattr(agent,'act') else int(agent(state))
        state,reward,done,truncated,info=env.step(action)
        records.append(dict(action=ACTION_NAMES[action],reward=reward,
                            speed_m_s=info.get('speed',np.nan),crashed=info.get('crashed',False)))
        if len(frames)<max_frames:frames.append(PILImage.fromarray(env.render()))
    env.close()
    buffer=io.BytesIO()
    frames[0].save(buffer,format='GIF',save_all=True,append_images=frames[1:],duration=220,loop=0)
    if save_path is not None:
        Path(save_path).write_bytes(buffer.getvalue())
    display(Image(data=buffer.getvalue(),format='gif'))
    return pd.DataFrame(records)



class PlaybackDQN:
    """Inference only: no training functions or exercise solutions."""
    def __init__(self, path=None):
        # State keys retain the original net.* names for checkpoint compatibility.
        self.model=nn.Module()
        self.model.net=nn.Sequential(nn.Linear(25,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,5))
        path=Path(path or ROOT / 'assets/checkpoints/dqn_reference_20k_seed0.npz')
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['config'] != HIGHWAY_CONFIG:
            raise ValueError('Checkpoint environment does not match HIGHWAY_CONFIG.')
        import hashlib
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
            raise ValueError('Checkpoint checksum mismatch.')
        with np.load(path,allow_pickle=False) as data:
            self.model.load_state_dict({k:torch.from_numpy(data[k].copy()) for k in data.files})
        self.model.eval()
        self.rng=np.random.default_rng(0)

    def act(self,state,epsilon=0.):
        with torch.no_grad():
            values=self.model.net(torch.as_tensor(state,dtype=torch.float32)).numpy()
        return random_argmax(values,self.rng)

def load_cached_comparison():
    folder=ROOT / 'assets/reference_results'
    return (pd.read_csv(folder / 'highway_learning_curves.csv'),
            pd.read_csv(folder / 'highway_heldout_evaluation.csv'))
