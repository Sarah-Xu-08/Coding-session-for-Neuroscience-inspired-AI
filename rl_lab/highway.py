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
import torch
from torch import nn
from PIL import Image as PILImage
from IPython.display import display, Image
from . import ROOT

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

def make_highway(render=False):
    env=gym.make('highway-fast-v0',config=copy.deepcopy(HIGHWAY_CONFIG),
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
    for seed in seeds:
        if hasattr(agent,'rng'):agent.rng=np.random.default_rng(int(seed)+800000)
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
                         seen_fraction=seen/max(steps,1) if isinstance(agent,SparseTrafficQ) else np.nan,
                         lane_change_requests=changes))
    env.close()
    return pd.DataFrame(rows)

def train_highway(kind,rules,dqn_factory=None,steps=2000,seed=0,eval_every=None,eval_seeds=(81000,81001,81002),
                  device='cpu',verbose=True):
    env=make_highway()
    if kind == 'DQN' and dqn_factory is None:
        raise ValueError('Supply the DQNAgent class defined in your notebook.')
    agent=dqn_factory(seed=seed,device=device) if kind=='DQN' else SparseTrafficQ(rules,kind,seed=seed)
    eval_every=eval_every or max(steps//4,1)
    resets=np.random.default_rng(seed+50000)
    state,_=env.reset(seed=int(resets.integers(2**31)))
    logs=[];started=time.perf_counter();paused=0.
    checkpoints=sorted(set([0,*range(eval_every,steps+1,eval_every),steps]))
    def record(step):
        nonlocal paused
        # Evaluation must not advance the exploration RNG used during training.
        rng_state=copy.deepcopy(agent.rng.bit_generator.state)
        evaluation_started=time.perf_counter()
        frame=highway_evaluate(agent,eval_seeds)
        agent.rng.bit_generator.state=rng_state
        paused+=time.perf_counter()-evaluation_started
        logs.append(dict(step=step,return_=frame.return_.mean(),collision=frame.collision.mean(),
                         train_seconds=time.perf_counter()-started-paused))
        if verbose: print(f'{kind:>6} seed={seed} step={step:>6,} '
                          f'eval return={frame.return_.mean():5.1f} '
                          f'collisions={frame.collision.mean():.0%}',flush=True)
    record(0)
    for step in range(1,steps+1):
        epsilon=float(np.interp(step,[0,steps*.7],[1.,.05]))
        action=agent.act(state,epsilon)
        nxt,reward,terminated,truncated,info=env.step(action)
        if kind=='DQN':
            agent.buffer.add(state,action,reward,nxt,terminated)
            if step>=256 and step%4==0:agent.learn()
            if step%400==0:agent.sync_target()
        else:
            agent.update(state,action,reward,nxt,terminated)
        state=nxt
        if terminated or truncated:
            state,_=env.reset(seed=int(resets.integers(2**31)))
        if step in checkpoints:record(step)
    env.close()
    agent.training_steps=steps
    agent.training_seconds=time.perf_counter()-started-paused
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
    palette={'coarse':'#438c68','medium':'#2563a6','fine':'#8b63b4','DQN':'#db7130'}
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for kind,frame in curves.groupby('method',sort=False):
        agg=frame.groupby('step')['return_'].agg(['mean','min','max'])
        axes[0].plot(agg.index,agg['mean'],label=kind,color=palette[kind])
        if frame.seed.nunique()>1:
            axes[0].fill_between(agg.index,agg['min'],agg['max'],alpha=.15,color=palette[kind])
        times=frame.groupby('step').train_seconds.mean()
        axes[1].plot(times.loc[agg.index],agg['mean'],label=kind,color=palette[kind])
    axes[0].set(xlabel='Training environment interactions',ylabel='Mean evaluation return',
                title='Same interaction budget')
    axes[1].set(xlabel='Training wall time (seconds; evaluation excluded)',ylabel='Mean evaluation return',
                title='Computation also has a cost')
    for ax in axes:ax.legend()
    if save_dir is not None:
        fig.savefig(Path(save_dir) / 'highway_comparison.png', dpi=180)
    plt.show()
    summary=scores.groupby('method',sort=False).agg(
        mean_return=('return_','mean'),collision_rate=('collision','mean'),
        mean_speed_m_s=('mean_speed_m_s','mean'),training_seconds=('train_seconds','mean'),
        table_states=('table_states','mean'),eval_state_coverage=('seen_fraction','mean'))
    display(summary.round(3))
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
