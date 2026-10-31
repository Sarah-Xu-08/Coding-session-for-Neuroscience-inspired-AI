"""Adapters to the unchanged upstream games and optional independent learners."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#@title Existing-game adapters and small independent Q-learning agents
def make_social_game(name,nrounds=20):
    from pettingzoo_dilemmas import prisoners_dilemma_v0
    if name != 'prisoners_dilemma':
        raise ValueError('This lesson uses Prisoner\'s Dilemma; Simple Spread is in rl_lab.spread.')
    return prisoners_dilemma_v0.env(nrounds=nrounds)

def read_payoffs(name):
    """Read all four payoff pairs by stepping the existing upstream environment."""
    result=np.zeros((2,2,2),float)
    game=make_social_game(name,nrounds=1)
    for a in range(2):
        for b in range(2):
            game.reset()
            players=game.possible_agents
            rewards=game.step({players[0]:a,players[1]:b})[1]
            result[a,b]=[rewards[p] for p in players]
    game.close()
    return result

def plot_payoffs(name):
    matrix=read_payoffs(name)
    names=['Cooperate','Defect']
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
    ax.imshow(matrix.sum(axis=-1),cmap='Blues',alpha=.5)
    for a in range(2):
        for b in range(2):
            ax.text(b,a,f'({matrix[a,b,0]:g}, {matrix[a,b,1]:g})',ha='center',va='center',fontsize=15)
    ax.set_xticks([0,1],names);ax.set_yticks([0,1],names)
    ax.set(xlabel='Player B',ylabel='Player A',title=name.replace('_',' ').title()+' · (A, B) payoffs')
    plt.show()
    return matrix

class SocialQ:
    def __init__(self,rules,seed=0,alpha=.15,gamma=.95,initial_preference=None):
        self.rules=rules
        self.q={};self.rng=np.random.default_rng(seed)
        self.alpha,self.gamma=alpha,gamma
        self.initial=np.zeros(2)
        if initial_preference is not None:self.initial[initial_preference]=2.

    def values(self,state,create=False):
        if create and state not in self.q:self.q[state]=self.initial.copy()
        return self.q.get(state,self.initial)

    def act(self,state,epsilon=0.):
        return self.rules.epsilon_greedy(self.values(state),epsilon,self.rng)

    def update(self,state,action,reward,next_state,done):
        row=self.values(state,True)
        target=self.rules.td_target(reward,done,self.values(next_state),np.ones(2,bool),self.gamma)
        row[action]=self.rules.q_update(row[action],target,self.alpha)

def strategy_action(policy,state,epsilon=0.):
    if isinstance(policy,SocialQ):return policy.act(state,epsilon)
    if policy == 'always_cooperate':return 0
    if policy == 'always_defect':return 1
    if callable(policy):return int(policy(state[2]))
    raise ValueError('Unknown social strategy: '+str(policy))

def social_match(name,policy_a,policy_b,nrounds=20,epsilon=0.,learn=False):
    game=make_social_game(name,nrounds)
    game.reset()
    players=game.possible_agents
    previous=(2,2)  # 2 is the explicit "no previous action" marker.
    records=[]
    for turn in range(nrounds):
        remaining=nrounds-turn
        state_a=(remaining,previous[0],previous[1])
        state_b=(remaining,previous[1],previous[0])
        # Both decisions depend only on PAST actions, before either is submitted.
        a=strategy_action(policy_a,state_a,epsilon)
        b=strategy_action(policy_b,state_b,epsilon)
        _,rewards,_,_,_=game.step({players[0]:a,players[1]:b})
        ra,rb=[rewards[p] for p in players]
        done=turn==nrounds-1
        if learn:
            if isinstance(policy_a,SocialQ):policy_a.update(state_a,a,ra,(remaining-1,a,b),done)
            if isinstance(policy_b,SocialQ):policy_b.update(state_b,b,rb,(remaining-1,b,a),done)
        records.append(dict(round=turn+1,a=a,b=b,reward_a=ra,reward_b=rb,
                            both_cooperate=int(a==0 and b==0)))
        previous=(a,b)
    # The upstream version reports truncation on one extra empty step. Stop
    # after exactly nrounds scored decisions; do not train on that zero-payoff call.
    game.close()
    return pd.DataFrame(records)

def train_social(name,rules,episodes=500,opponent='learner',seed=0,
                 initial_preference=None,nrounds=20,epsilon_start=.5):
    a=SocialQ(rules,seed=seed,initial_preference=initial_preference)
    b=SocialQ(rules,seed=seed+100,initial_preference=initial_preference) if opponent=='learner' else opponent
    logs=[]
    for episode in range(episodes):
        epsilon=float(np.interp(episode,[0,episodes*.8],[epsilon_start,.03]))
        frame=social_match(name,a,b,nrounds,epsilon,learn=True)
        logs.append(dict(episode=episode+1,cooperation=frame.both_cooperate.mean(),
                         reward_a=frame.reward_a.mean(),reward_b=frame.reward_b.mean()))
    return a,b,pd.DataFrame(logs)

def plot_social(history,match,name):
    labels=['Cooperate','Defect']
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
    sm=history.cooperation.rolling(min(50,len(history)),min_periods=1).mean()
    axes[0].plot(history.episode,sm)
    axes[0].set(xlabel='Training sessions',ylabel='Fraction of rounds',ylim=(-.02,1.02),
                title='Mutual cooperation · rolling mean')
    counts=np.zeros((2,2))
    for a,b in zip(match.a,match.b):counts[a,b]+=1
    counts/=len(match)
    axes[1].imshow(counts,vmin=0,vmax=1,cmap='Blues',alpha=.6)
    for a in range(2):
        for b in range(2):axes[1].text(b,a,f'{counts[a,b]:.0%}',ha='center',va='center')
    axes[1].set_xticks([0,1],labels);axes[1].set_yticks([0,1],labels)
    axes[1].set(xlabel='Player B',ylabel='Player A',title='One greedy evaluation session')
    plt.show()
