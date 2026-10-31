"""Four editable teaching illustrations; conceptual scenes, not measured results.

All artwork is generated here; no external image assets are required.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import rl_lab
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse, Polygon, FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import to_rgb
from rl_lab.social import read_payoffs

INK, MUTED = '#243047', '#536175'
BLUE, PINK, YELLOW, GREEN = '#238be6', '#ee6d9f', '#ffc654', '#32b592'


def text(ax, x, y, value, size=17, color=INK, **kwargs):
    return ax.text(x, y, value, fontsize=size, color=color, fontfamily='DejaVu Sans', **kwargs)


def box(ax, x, y, w, h, color, radius=.15, edge='none', lw=1, z=1, alpha=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                       fc=color, ec=edge, lw=lw, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def arrow(ax, start, end, color=INK, curve=0., lw=3, scale=18, z=3):
    p = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=scale,
                        connectionstyle=f'arc3,rad={curve}', color=color, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def canvas(title, subtitle, top, bottom):
    fig, ax = plt.subplots(figsize=(16,9))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0,16), ylim=(0,9), aspect='equal')
    ax.axis('off')
    a, b = np.array(to_rgb(top)), np.array(to_rgb(bottom))
    v = np.linspace(0, 1, 400)[:, None, None]
    ax.imshow(a[None, None, :] * v + b[None, None, :] * (1-v), extent=[0,16,0,9],
              origin='lower', aspect='auto', zorder=-10)
    text(ax,.6,8.28,title,size=32,weight='bold')
    text(ax,.62,7.74,subtitle,size=17,color=MUTED)
    return fig, ax


def loop(ax, observe, act, receive, accent=BLUE):
    for x, key, value in [(0.65,'OBSERVE',observe),(5.85,'ACT',act),(11.05,'RECEIVE',receive)]:
        box(ax,x-.15,.3,4.95,1.05,'white',radius=.18,alpha=.9)
        ax.add_patch(Circle((x+.06,1.04),.055,fc=accent,zorder=3))
        text(ax,x+.22,.98,key,size=11,color=MUTED,weight='bold')
        text(ax,x+.02,.63,value,size=15,va='center')


def cookie(ax, x, y, r=.3):
    ax.add_patch(Ellipse((x,y-.07),r*2.2,r*.55,fc='#a86d45',alpha=.18,zorder=4))
    ax.add_patch(Circle((x,y),r,fc='#d99a54',ec='#b9783a',lw=2,zorder=5))
    ax.add_patch(Circle((x-.03,y+.03),r*.8,fc='#edb569',zorder=6))
    for dx,dy in [(-.45,.3),(.35,.4),(.4,-.3),(-.25,-.45),(.05,-.04)]:
        ax.add_patch(Circle((x+dx*r,y+dy*r),r*.095,fc='#654030',zorder=7))


def cake(ax, x, y, scale=1):
    ax.add_patch(Ellipse((x+.18*scale,y-.08*scale),.95*scale,.2*scale,fc='#b0c6a4',alpha=.25,zorder=4))
    ax.add_patch(Polygon(np.array([[0,0],[.7,0],[.7,.62],[0,.38]])*scale+[x,y],
                          fc='#9bc576',ec='#6d9a54',lw=2,zorder=5))
    ax.add_patch(Polygon(np.array([[0,.38],[.7,.62],[.3,.86]])*scale+[x,y],
                          fc='#d5e9a5',ec='#6d9a54',lw=2,zorder=6))
    ax.plot([x,x+.7*scale],[y+.15*scale,y+.15*scale],color='#eff0ce',lw=4,zorder=7)
    for dx,dy in [(.28,.64),(.39,.59),(.24,.55)]:
        ax.add_patch(Ellipse((x+dx*scale,y+dy*scale),.075*scale,.04*scale,angle=25,fc='#698c45',zorder=7))


def person(ax, x, y, color, scale=1):
    ax.add_patch(Ellipse((x,y-.12*scale),.7*scale,.18*scale,fc=INK,alpha=.1,zorder=1))
    box(ax,x-.24*scale,y,.48*scale,.7*scale,color,radius=.16*scale,z=4)
    for dx in [-.12,.12]:
        ax.plot([x+dx*scale,x+dx*scale],[y-.01*scale,y-.38*scale],color=INK,lw=4,zorder=3)
    ax.add_patch(Circle((x,y+1.0*scale),.27*scale,fc='#f3bd92',zorder=5))
    ax.add_patch(Ellipse((x-.01*scale,y+1.15*scale),.53*scale,.26*scale,fc='#624637',zorder=6))
    for dx in [-.09,.09]:
        ax.add_patch(Circle((x+dx*scale,y+1.0*scale),.025*scale,fc=INK,zorder=7))


def cafe():
    fig,ax=canvas('DESSERT CAFÉ', 'Prepare ahead: today’s action changes next hour’s stock.', '#fff6e3','#ffe8cc')
    box(ax,.65,1.65,9.0,5.6,'#ffdda8',radius=.3)
    box(ax,.85,1.67,8.6,.16,'#d89b67',radius=.05)
    box(ax,1.2,5.52,7.9,1.25,'#fff9ef',radius=.2)
    text(ax,5.15,6.17,'THE LEARNING CAFÉ',size=24,ha='center',weight='bold')
    for i in range(12):
        x=1.06+i*.68
        c=PINK if i%2==0 else '#fff6e8'
        ax.add_patch(Rectangle((x,4.92),.68,.55,fc=c,zorder=3))
        ax.add_patch(Ellipse((x+.34,4.92),.68,.3,fc=c,zorder=3))
    box(ax,1.17,2.25,5.5,2.05,'#ddf5f3',radius=.12,edge='#9bcfc6',lw=2)
    ax.plot([1.3,6.5],[3.2,3.2],color='#8ec8bf',lw=3)
    box(ax,1.02,1.87,5.8,.45,'#bb7352',radius=.06,z=8)
    for x in [1.8,2.55,3.3]:cookie(ax,x,3.65,.29)
    for x in [4.35,5.35]:cake(ax,x,3.34,.75)
    text(ax,2.6,2.61,'COOKIE · 6 credits',size=13,ha='center',weight='bold')
    text(ax,5.05,2.61,'CAKE · 7 credits',size=13,ha='center',weight='bold')
    person(ax,7.65,2.65,BLUE,.85);person(ax,8.72,2.45,GREEN,.67)
    box(ax,10.2,4.4,5.2,2.45,'white',radius=.25)
    box(ax,10.55,4.72,1.55,1.48,'#f49a64',radius=.15)
    box(ax,10.73,4.88,1.18,.85,'#5c5664',radius=.08)
    for x in np.linspace(10.83,11.83,6):ax.add_patch(Circle((x,6.0),.045,fc=INK))
    cookie(ax,11.12,5.22,.17);cookie(ax,11.6,5.22,.17)
    arrow(ax,(12.4,5.4),(13.15,5.4),color=PINK)
    cookie(ax,13.65,5.34,.24);cake(ax,14.3,5.05,.62)
    text(ax,12.8,6.43,'PREPARE NOW → STOCK NEXT HOUR',size=12,ha='center',weight='bold')
    text(ax,12.8,4.56,'New preparation is not sold immediately.',size=12,ha='center',color=MUTED)
    ax.add_patch(Circle((11.0,2.85),.62,fc='white',ec='#e1b681',lw=2))
    ax.plot([11,11,11.36],[3.23,2.85,2.85],color=INK,lw=3,solid_capstyle='round')
    text(ax,12,3.27,'10 a.m. – 6 p.m.',size=19,weight='bold')
    text(ax,12,2.76,'24 prep units · limited shelf space',size=12,color=MUTED)
    text(ax,12,2.3,'Balance sales, waste and missed orders.',size=12,color=MUTED)
    loop(ax,'Hour, stock, ingredients','Choose a preparation batch','Profit − service penalty',PINK)
    return fig


def car(ax,x,y,color,scale=1,ego=False):
    box(ax,x-.35*scale,y-.73*scale,.8*scale,1.55*scale,'#172b3a',radius=.15,z=4,alpha=.18)
    for dx in [-.4,.35]:
        for dy in [-.43,.38]:box(ax,x+dx*scale,y+dy*scale,.13*scale,.3*scale,'#26303b',radius=.035,z=5)
    box(ax,x-.35*scale,y-.65*scale,.76*scale,1.42*scale,color,radius=.17,z=6)
    box(ax,x-.25*scale,y+.23*scale,.56*scale,.31*scale,'#d5f2ff',radius=.06,z=7)
    box(ax,x-.25*scale,y-.43*scale,.56*scale,.2*scale,'#99cde3',radius=.04,z=7)
    ax.plot([x-.27*scale,x+.31*scale],[y+.63*scale,y+.63*scale],color='#fff5be',lw=3,zorder=8)
    if ego:text(ax,x+.025*scale,y-.04*scale,'YOU',size=10,color='white',ha='center',weight='bold',zorder=9)


def highway():
    fig,ax=canvas('HIGHWAY NAVIGATION', 'Observe traffic, compare five action values, then choose a move.', '#e6f6ff','#dcf1e5')
    box(ax,.65,1.7,9.15,5.55,'#9ecb9a',radius=.3)
    ax.add_patch(Rectangle((2,1.7),6.25,5.55,fc='#455367',zorder=1))
    for x in [2.13,8.12]:ax.plot([x,x],[1.7,7.25],color=YELLOW,lw=3,zorder=2)
    for x in [4.1,6.17]:
        for y in np.arange(1.88,7.0,.85):ax.plot([x,x],[y,min(y+.46,7.25)],color='#ebf5f9',lw=3,zorder=2)
    for x,y in [(1.1,3),(8.98,4.9),(1.2,6.2),(9.0,2.5)]:
        ax.add_patch(Circle((x,y),.27,fc='#67a978',zorder=2))
        ax.add_patch(Circle((x-.06,y+.07),.2,fc='#83bf80',zorder=3))
    car(ax,5.12,3.0,BLUE,ego=True);car(ax,5.12,6.05,PINK)
    car(ax,7.15,4.95,'#b5c1d1');car(ax,3.1,6.2,YELLOW);car(ax,3.1,2.8,'#8dcbb9')
    arrow(ax,(4.94,3.88),(3.14,4.72),color='#98e5fa',curve=-.3,lw=4,scale=23,z=10)
    text(ax,5.15,1.9,'Schematic scene · not a policy result',size=11,color='white',ha='center',zorder=9)
    box(ax,10.25,4.66,5.1,2.25,'white',radius=.25)
    text(ax,10.58,6.42,'WHAT THE AGENT OBSERVES',size=12,weight='bold',color=MUTED)
    text(ax,10.58,5.88,'5 vehicles × 5 features',size=21,weight='bold')
    text(ax,10.58,5.38,'Presence, position and velocity',size=14)
    text(ax,10.58,4.96,'25 numbers → a Q-table or Q-network',size=12,color=MUTED)
    text(ax,10.45,4.06,'FIVE POSSIBLE ACTIONS',size=12,weight='bold',color=MUTED)
    for x,y,label,c in [(10.45,3.16,'Lane left',BLUE),(12.95,3.16,'Lane right',BLUE),
                         (10.45,2.42,'Faster',GREEN),(12.95,2.42,'Slower',PINK)]:
        box(ax,x,y,2.15,.55,c,radius=.12)
        text(ax,x+1.075,y+.275,label,size=14,color='white',ha='center',va='center',weight='bold')
    text(ax,12.7,1.91,'…or maintain lane and speed',size=13,ha='center',color=MUTED)
    loop(ax,'Nearby traffic: 25 features','Lane / speed / maintain','Speed reward; crash penalty',BLUE)
    return fig


def robot(ax,x,y,color,scale=1,label=None):
    ax.add_patch(Ellipse((x,y-.4*scale),.9*scale,.18*scale,fc=INK,alpha=.1,zorder=3))
    box(ax,x-.29*scale,y-.25*scale,.58*scale,.55*scale,color,radius=.12*scale,z=5)
    ax.add_patch(Circle((x,y+.52*scale),.43*scale,fc=color,ec='white',lw=2,zorder=6))
    box(ax,x-.3*scale,y+.37*scale,.6*scale,.29*scale,'#243047',radius=.11*scale,z=7)
    for dx in [-.14,.14]:ax.add_patch(Circle((x+dx*scale,y+.53*scale),.055*scale,fc='#d5f7ff',zorder=8))
    ax.plot([x,x],[y+.94*scale,y+1.11*scale],color=INK,lw=2,zorder=6)
    ax.add_patch(Circle((x,y+1.15*scale),.07*scale,fc=YELLOW,zorder=7))
    if label is not None:text(ax,x,y-.08*scale,label,size=12*scale,color='white',ha='center',weight='bold',zorder=8)


def dilemma():
    fig,ax=canvas('PRISONER’S DILEMMA', 'Same round. Simultaneous choices. Separate rewards.', '#f4edff','#e9f5ff')
    robot(ax,2.25,4.45,BLUE,1.38,'A');robot(ax,5.5,4.45,PINK,1.38,'B')
    text(ax,2.25,6.6,'PLAYER A',size=17,weight='bold',ha='center',color=BLUE)
    text(ax,5.5,6.6,'PLAYER B',size=17,weight='bold',ha='center',color='#c54b7b')
    for x in [1.08,4.33]:
        box(ax,x,3.0,1.1,.64,GREEN,radius=.13);box(ax,x+1.3,3.0,1.1,.64,'#ee8d70',radius=.13)
        text(ax,x+.55,3.32,'C',size=22,ha='center',va='center',weight='bold',color='white')
        text(ax,x+1.85,3.32,'D',size=22,ha='center',va='center',weight='bold',color='white')
    text(ax,3.86,2.47,'C = cooperate     D = defect',size=15,ha='center')
    text(ax,3.86,1.9,'Use past rounds, not the partner’s current choice.',size=12,color=MUTED,ha='center')
    arrow(ax,(7.05,4.7),(8.0,4.7),color='#9b8bc9',lw=3,scale=25)
    matrix=read_payoffs('prisoners_dilemma')
    x0,y0,w,h=9.7,2.7,4.65,3.35
    text(ax,12.0,6.96,'PLAYER B CHOOSES',size=12,ha='center',weight='bold',color=MUTED)
    for b,name in enumerate(['Cooperate','Defect']):text(ax,x0+(b+.5)*w/2,6.35,name,size=16,ha='center',weight='bold')
    for a,name in enumerate(['Cooperate','Defect']):
        text(ax,9.4,y0+(1.5-a)*h/2,name,size=14,ha='right',va='center',weight='bold')
        for b in range(2):
            x=x0+b*w/2;y=y0+(1-a)*h/2
            c=[['#b7e9d6','#f8ddbd'],['#f8ddbd','#d8d4f3']][a][b]
            box(ax,x+.025,y+.025,w/2-.05,h/2-.05,c,radius=.16)
            ra,rb=matrix[a,b]
            text(ax,x+w/4,y+h/4,f'({ra:g}, {rb:g})',size=27,ha='center',va='center',weight='bold')
    text(ax,12,2.19,'Each cell = (A’s payoff, B’s payoff)',size=14,ha='center')
    text(ax,12,1.75,'Mutual cooperation helps both; defection tempts each.',size=11,color=MUTED,ha='center')
    loop(ax,'Partner’s previous action','Cooperate or defect','Your own payoff each round',PINK)
    return fig


def flag(ax,x,y,label):
    ax.add_patch(Circle((x,y),.26,fc='#81d5ac',alpha=.35,zorder=1))
    ax.add_patch(Circle((x,y),.10,fc='#267959',zorder=2))
    ax.plot([x,x],[y,y+.75],color='#267959',lw=2,zorder=3)
    ax.add_patch(Polygon([[x,y+.75],[x+.42,y+.61],[x,y+.45]],fc=GREEN,zorder=4))
    text(ax,x+.12,y+.83,label,size=13,color='#20634c',ha='center',weight='bold')


def spread():
    fig,ax=canvas('SIMPLE SPREAD', 'Three agents. Three landmarks. One identical team reward.', '#e5f8f0','#e1f1ff')
    for x in [.65,8.3]:box(ax,x,1.75,7.05,5.5,'#f7fcf3',radius=.28)
    text(ax,4.15,6.79,'EVERYONE CHOOSES NEAREST',size=17,ha='center',weight='bold')
    text(ax,11.85,6.79,'COVER DIFFERENT LANDMARKS',size=17,ha='center',weight='bold')
    for offset in [0,7.65]:
        for x,y,label in [(2.4,3.1,'L1'),(6.35,3.1,'L2'),(4.4,5.45,'L3')]:flag(ax,x+offset,y,label)
    for x,y,c,label in [(1.7,2.85,BLUE,'0'),(3.05,3.0,PINK,'1'),(1.55,4.4,'#e8a02c','2')]:
        robot(ax,x,y,c,.5,label)
        arrow(ax,(x,y-.06),(2.4,3.1),color=c,lw=2,scale=13,z=2)
    for x,y,c,label in [(10.05,3.1,BLUE,'0'),(14.0,3.1,PINK,'1'),(12.05,5.45,'#e8a02c','2')]:robot(ax,x,y,c,.53,label)
    text(ax,6.3,2.34,'unattended',size=12,color='#b64c54',ha='center')
    text(ax,5.55,5.18,'unattended',size=12,color='#b64c54',ha='center')
    text(ax,4.1,1.99,'Duplicated effort leaves gaps.',size=14,ha='center',color=MUTED)
    text(ax,11.85,1.99,'Different roles improve team coverage.',size=14,ha='center',color=MUTED)
    box(ax,7.15,4.15,1.7,1.08,YELLOW,radius=.2,z=9)
    text(ax,8,4.83,'ONE SCORE',size=11,ha='center',weight='bold',zorder=10)
    text(ax,8,4.46,'r₀ = r₁ = r₂',size=15,ha='center',weight='bold',zorder=10)
    loop(ax,'Positions of targets + teammates','Stay / left / right / down / up','Shared coverage reward',GREEN)
    return fig


def generate():
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    folder=ROOT/'assets/figures';folder.mkdir(parents=True,exist_ok=True)
    for name,draw in [('cafe',cafe),('highway',highway),('prisoners_dilemma',dilemma),('simple_spread',spread)]:
        fig=draw()
        for suffix in ['png','svg']:fig.savefig(folder/f'task_{name}.{suffix}',dpi=130)
        plt.close(fig)
        print(f'Generated task_{name}.png and .svg')


if __name__=='__main__':generate()
