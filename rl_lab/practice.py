"""Optional notebook controls. Every control has an ordinary-code alternative."""
import numpy as np
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
from .cafe import DessertCafe
from .social import social_match, read_payoffs, plot_stag_threshold

def cafe_controls(seed=2026):
    env=DessertCafe()
    output=widgets.Output()
    action=widgets.Dropdown(description='Prepare:')
    advance=widgets.Button(description='Serve this hour',button_style='primary')
    reset=widgets.Button(description='Reset same day')
    total=0.
    def show():
        mask=env.action_mask()
        valid=[int(a) for a in np.flatnonzero(mask)]
        previous_action=action.value
        action.options=[(str(tuple(env.ACTIONS[a])),a) for a in valid]
        action.value=previous_action if previous_action in valid else valid[0]
        advance.disabled=env.t>=env.horizon
        with output:
            clear_output(wait=True)
            if env.t < env.horizon:
                print(f'{10+env.t}:00 | stock {tuple(env.stock)} | ingredients {env.ingredients}')
                print('Choose before seeing this hour’s arrivals. Preparation is available NEXT hour.')
            else:print('Closed. Reset to try the same customer day again.')
            print(f'Cumulative reward: {total:.1f} credits')
    def step(_):
        nonlocal total
        if env.t>=env.horizon:return
        _,reward,done,_,info=env.step(action.value)
        total+=reward
        show()
        with output:
            print(f'Completed hour: demand {tuple(info["demand"])}, sales {tuple(info["sales"])}')
            print(f'Prepared {tuple(info["produced"])}, missed {tuple(info["missed"])}, waste {tuple(info["waste"])}')
            print(f'Cash profit this hour: {info["cash_profit"]:.1f}; reward: {reward:.1f}')
    def restart(_=None):
        nonlocal total
        env.reset(seed=seed);total=0.;show()
    advance.on_click(step);reset.on_click(restart);restart()
    display(widgets.VBox([widgets.HBox([action,advance,reset]),output]))
    return {'env':env,'action':action,'advance':advance,'reset':reset}

def dilemma_controls(tit_for_tat):
    options=[('Always cooperate','always_cooperate'),('Always defect','always_defect'),('Tit-for-tat',tit_for_tat)]
    a=widgets.Dropdown(options=options,description='Player A:')
    b=widgets.Dropdown(options=options,description='Player B:')
    run=widgets.Button(description='Play 10 rounds',button_style='primary')
    output=widgets.Output()
    def play(_):
        with output:
            clear_output(wait=True)
            frame=social_match('prisoners_dilemma',a.value,b.value,nrounds=10)
            display(frame)
            print('Mean payoffs:',frame[['reward_a','reward_b']].mean().round(2).to_dict())
    run.on_click(play)
    display(widgets.VBox([widgets.HBox([a,b,run]),output]))
    return {'a':a,'b':b,'run':run}

def stag_controls(expected_payoffs):
    matrix=read_payoffs('stag_hunt')
    slider=widgets.FloatSlider(value=.5,min=0,max=1,step=.05,description='Partner p:',continuous_update=False)
    output=widgets.Output()
    def draw(change=None):
        with output:
            clear_output(wait=True)
            values=expected_payoffs(matrix,slider.value)
            print(f'Expected payoff: stag={values[0]:.2f}, hare={values[1]:.2f}')
            plot_stag_threshold(matrix,expected_payoffs,p_marker=slider.value)
    slider.observe(draw,names='value');draw()
    display(widgets.VBox([slider,output]))
    return {'slider':slider}
