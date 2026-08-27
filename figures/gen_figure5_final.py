import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size':11,'figure.dpi':300})
OUT = os.path.dirname(os.path.abspath(__file__))

C = [
 dict(t=1.5, geo='S=4.0, d=3\n$\\varphi$=29.9%',  nemd=1.390, ns=0.010, ml=1.341, ms=0.125,
      seeds=[1.40,1.39,1.39],  c='#2E7D5B'),
 dict(t=2.0, geo='S=3.5, d=3\n$\\varphi$=22.4%',  nemd=2.091, ns=0.048, ml=2.116, ms=0.131,
      seeds=[2.152,2.035,2.086], c='#2E86AB'),
 dict(t=5.0, geo='S=2.5, d=1\n$\\varphi$=11.8%',  nemd=5.079, ns=0.483, ml=4.728, ms=0.292,
      seeds=[5.186,5.878,4.986,5.025,4.223,5.175], c='#C44E52'),
]
x = np.arange(len(C)); w = 0.36
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

a = ax[0]
a.bar(x-w/2, [c['nemd'] for c in C], w, yerr=[c['ns'] for c in C], capsize=4,
      color='#E8756D', edgecolor='k', lw=.6, label='NEMD')
a.bar(x+w/2, [c['ml'] for c in C], w, yerr=[c['ms'] for c in C], capsize=4,
      color='#6BA6CD', edgecolor='k', lw=.6, label='GP prediction')
for i,c in enumerate(C):
    n = abs(c['nemd']-c['ml'])/np.hypot(c['ns'],c['ms'])
    top = max(c['nemd']+c['ns'], c['ml']+c['ms'])
    a.text(i, top*1.16, f'{n:.1f}$\\sigma$', ha='center',
           fontsize=10, fontweight='bold', color='#2B7A3E' if n<=1 else '#B5651D')
    a.text(i-w/2, c['nemd']+c['ns'], f"{c['nemd']:.2f}", ha='center', va='bottom', fontsize=8.5)
    a.text(i+w/2, c['ml']+c['ms'],  f"{c['ml']:.2f}",  ha='center', va='bottom', fontsize=8.5)
a.set_xticks(x); a.set_xticklabels([f"$\\kappa_{{\\rm target}}$ = {c['t']}\n{c['geo']}" for c in C], fontsize=9)
a.set_ylabel('$\\kappa$ (W m$^{-1}$K$^{-1}$)'); a.set_ylim(0, 7.2)
a.set_title('(a) NEMD vs GP prediction', fontweight='bold', fontsize=11)
a.legend(fontsize=9, loc='upper left'); a.grid(alpha=.25, ls=':', axis='y')

b = ax[1]
for i,c in enumerate(C):
    b.fill_between([i-.32,i+.32], c['ml']-c['ms'], c['ml']+c['ms'], color=c['c'], alpha=.18)
    b.plot([i-.32,i+.32], [c['ml']]*2, '--', color=c['c'], lw=1.4)
    b.scatter(np.linspace(i-.18,i+.18,len(c['seeds'])), c['seeds'],
              color=c['c'], s=55, ec='k', lw=.5, zorder=3)
b.set_xticks(x); b.set_xticklabels([f"$\\kappa_{{\\rm target}}$ = {c['t']}" for c in C])
b.set_ylabel('$\\kappa$ (W m$^{-1}$K$^{-1}$)'); b.set_xlim(-.6, len(C)-.4); b.set_ylim(0, 6.6)
b.set_title('(b) Individual seeds vs GP $\\pm1\\sigma$', fontweight='bold', fontsize=11)
b.legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',
                  markeredgecolor='k',markersize=8,label='NEMD seed'),
                  Line2D([0],[0],ls='--',color='gray',label='GP prediction')],
         fontsize=8.5, loc='upper left'); b.grid(alpha=.25, ls=':')

fig.tight_layout()
fig.savefig(f'{OUT}/fig_validation_final.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(f'{OUT}/fig_validation_final.pdf', bbox_inches='tight', facecolor='white')
print('saved fig_validation_final.png + .pdf')
