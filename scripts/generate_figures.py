"""Rebuild the four-task overview as an editable SVG and a video-ready PNG."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rl_lab
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon
from rl_lab.social import read_payoffs

INK = '#172b45'
MUTED = '#53657a'
BLUE = '#2675bc'
ORANGE = '#d88735'
GREEN = '#328273'
PURPLE = '#8866b3'


def label(ax, x, y, text, size=12, color=INK, **kwargs):
    return ax.text(x, y, text, fontsize=size, color=color, transform=ax.transAxes, **kwargs)


def panel(ax, number, title, subtitle, color):
    ax.set(xlim=(0,1), ylim=(0,1))
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((.005,.005),.99,.99,boxstyle='round,pad=0,rounding_size=.025',
                              facecolor='white',edgecolor='#dce3eb',lw=1.1,transform=ax.transAxes))
    ax.add_patch(Circle((.055,.92),.027,facecolor=color,transform=ax.transAxes))
    label(ax,.055,.92,str(number),size=12,color='white',ha='center',va='center',weight='bold')
    label(ax,.1,.92,title,size=18,weight='bold',va='center')
    label(ax,.045,.84,subtitle,size=11,color=MUTED)


def footer(ax, observation, decision, payoff):
    ax.plot([.04,.96],[.245,.245],color='#e3e9ef',transform=ax.transAxes)
    for y, key, value in [(.19,'OBSERVE',observation),(.125,'DECIDE',decision),(.06,'RECEIVE',payoff)]:
        label(ax,.045,y,key,size=8.5,color=MUTED,weight='bold')
        label(ax,.205,y,value,size=10.5)


def matrix(ax, matrix, names, color, title=None):
    x0,y0,w,h=.33,.36,.56,.35
    for a in range(2):
        for b in range(2):
            x=x0+b*w/2;y=y0+(1-a)*h/2
            ax.add_patch(Rectangle((x,y),w/2,h/2,facecolor=color if a==b==0 else '#f0f4f8',
                                   alpha=.17 if a==b==0 else 1,edgecolor='white',lw=2,transform=ax.transAxes))
            pair=matrix[a,b]
            label(ax,x+w/4,y+h/4,f'({pair[0]:g}, {pair[1]:g})',size=15,ha='center',va='center',weight='bold')
    for i,name in enumerate(names):
        label(ax,x0+(i+.5)*w/2,.735,name,size=10.5,ha='center')
        label(ax,.28,y0+(1.5-i)*h/2,name,size=10.5,ha='right',va='center')
    label(ax,.61,.79,'PLAYER B',size=8.5,color=MUTED,ha='center',weight='bold')
    label(ax,.065,.535,'PLAYER A',size=8.5,color=MUTED,rotation=90,va='center',weight='bold')
    label(ax,.61,.29,'Each cell = (A payoff, B payoff)',size=10,color=MUTED,ha='center')


def generate():
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    fig, axes=plt.subplots(2,2,figsize=(15.5,10.7))
    fig.patch.set_facecolor('#f3f6fa')
    fig.subplots_adjust(left=.035,right=.965,top=.88,bottom=.055,hspace=.08,wspace=.045)
    fig.text(.04,.958,'FOUR TASKS. ONE DECISION LOOP.',fontsize=24,weight='bold',color=INK)
    fig.text(.04,.92,'Observe a situation → choose an action → receive reward → update what you expect',fontsize=13,color=MUTED)

    ax=axes[0,0]
    panel(ax,1,'Dessert café','A Q-table learns when to prepare each dessert.',BLUE)
    # Schematic shop, cookie and cake, followed by synthetic demand profiles.
    ax.add_patch(Rectangle((.07,.4),.25,.25,fc='#e8eff6',ec='#c6d4e2',transform=ax.transAxes))
    ax.add_patch(Polygon([[.05,.65],[.34,.65],[.29,.74],[.1,.74]],fc=BLUE,transform=ax.transAxes))
    label(ax,.195,.58,'CAFÉ',size=14,ha='center',weight='bold')
    ax.add_patch(Circle((.135,.46),.035,fc='#c59356',ec='#9e6d36',transform=ax.transAxes))
    for dx,dy in [(-.014,.01),(.01,-.014),(.015,.013)]:
        ax.add_patch(Circle((.135+dx,.46+dy),.004,fc='#61402c',transform=ax.transAxes))
    ax.add_patch(Polygon([[.22,.42],[.28,.42],[.28,.5],[.22,.47]],fc='#b3ca7c',ec='#6f8b44',transform=ax.transAxes))
    demand=ax.inset_axes([.43,.4,.49,.32])
    t=np.arange(10,18)
    from rl_lab.cafe import DessertCafe
    demand.plot(t,DessertCafe.MEANS[0],color=ORANGE,lw=2.2,label='Cookie')
    demand.plot(t,DessertCafe.MEANS[1],color=GREEN,lw=2.2,label='Cake')
    demand.set_xticks([10,12,14,16,17]);demand.set_xlabel('Hour',fontsize=9)
    demand.set_ylabel('Mean demand',fontsize=9);demand.tick_params(labelsize=8)
    demand.spines[['top','right']].set_visible(False)
    demand.legend(fontsize=8,frameon=False,loc='upper left')
    label(ax,.195,.3,'Prepare now → sell next hour',size=10,color=MUTED,ha='center')
    footer(ax,'Hour, stock, ingredients','How many cookies and cakes?','Profit − missed-service penalty')

    ax=axes[0,1]
    panel(ax,2,'Highway navigation','A Q-network maps 25 traffic features to 5 action values.',ORANGE)
    ax.add_patch(Rectangle((.07,.4),.53,.34,fc='#59697a',transform=ax.transAxes))
    for x in [.247,.423]:
        ax.plot([x,x],[.4,.74],ls=(0,(5,5)),color='white',lw=1.5,transform=ax.transAxes)
    def car(x,y,c):
        ax.add_patch(FancyBboxPatch((x,y),.068,.09,boxstyle='round,pad=0,rounding_size=.012',fc=c,ec='white',lw=1,transform=ax.transAxes))
        ax.add_patch(Rectangle((x+.009,y+.056),.05,.017,fc='#d9eaf4',transform=ax.transAxes))
    car(.31,.43,BLUE);car(.135,.62,'#c1ced8');car(.49,.55,'#c1ced8')
    ax.annotate('',xy=(.345,.65),xytext=(.345,.54),xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle='->',color='#c6e5ff',lw=2))
    label(ax,.345,.325,'Ego vehicle',size=10,color=BLUE,ha='center',weight='bold')
    for x,ys in [(.72,[.48,.57,.66]),(.82,[.48,.57,.66]),(.92,[.52,.62])]:
        for y in ys:
            ax.add_patch(Circle((x,y),.012,fc=ORANGE,transform=ax.transAxes))
    for x1,x2,ys1,ys2 in [(.72,.82,[.48,.57,.66],[.48,.57,.66]),(.82,.92,[.48,.57,.66],[.52,.62])]:
        for y1 in ys1:
            for y2 in ys2:ax.plot([x1+.013,x2-.013],[y1,y2],color='#d8b999',lw=.6,transform=ax.transAxes)
    label(ax,.82,.35,'25 → 64 → 64 → 5',size=10,color=MUTED,ha='center')
    footer(ax,'Nearby positions and velocities','Lane left/right, maintain, faster/slower','Speed/progress reward; crash penalty')

    ax=axes[1,0]
    panel(ax,3,"Prisoner’s Dilemma",'Individual temptation can undermine mutual cooperation.',GREEN)
    matrix(ax,read_payoffs('prisoners_dilemma'),['Cooperate','Defect'],GREEN)
    footer(ax,"Partner’s previous action",'Cooperate or defect','Your own payoff; future replies can change')

    ax=axes[1,1]
    panel(ax,4,'Stag Hunt','The best response depends on expecting coordination.',PURPLE)
    matrix(ax,read_payoffs('stag_hunt'),['Stag','Hare'],PURPLE)
    footer(ax,'Belief p that the partner chooses stag','Choose stag or hare','Your own payoff; coordination can pay more')
    fig.text(.04,.023,'Café data are synthetic. HighwayEnv is a simulation. Social payoffs come from the bundled upstream games.',fontsize=9,color=MUTED)
    folder=ROOT/'assets/figures';folder.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg']:
        fig.savefig(folder/f'four_tasks_overview.{suffix}',dpi=200,facecolor=fig.get_facecolor())
    for ax,name in zip(axes.flat,['cafe','highway','prisoners_dilemma','stag_hunt']):
        bounds=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(1.015,1.015)
        fig.savefig(folder/f'task_{name}.png',dpi=200,bbox_inches=bounds,facecolor=fig.get_facecolor())
    plt.close(fig)
    print('Generated assets/figures/four_tasks_overview.png and .svg')


if __name__=='__main__':generate()
