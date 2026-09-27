"""Render the report S1 schematic from the frozen 07b routing definition."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / 'docs' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(15, 9))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 15), ylim=(0, 9))
ax.axis('off')
ink, blue, teal, amber = '#233447', '#32628b', '#21796e', '#99702b'

def text(x, y, s, size=10, color=ink, weight='normal', ha='center'):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight,
            ha=ha, va='center', linespacing=1.45)

def box(x, y, w, h, title, body='', color=blue, fill='#f1f6fa', dashed=False):
    ax.add_patch(FancyBboxPatch((x,y), w,h, boxstyle='round,pad=0.02,rounding_size=0.09',
                 linewidth=1.15, edgecolor=color, facecolor=fill,
                 linestyle=(0,(5,3)) if dashed else '-'))
    text(x+w/2, y+h*(.67 if body else .5), title, 10.5, color, 'bold')
    if body: text(x+w/2, y+h*.29, body, 9)

def arrow(points, color=ink, dashed=False):
    for a,b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]], color=color, lw=1.15,
                linestyle='--' if dashed else '-')
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                  mutation_scale=11, lw=1.15, color=color,
                  linestyle='--' if dashed else '-'))

text(.45, 8.65, 'Alarm-preserving decision support for PHM2018', 18, weight='bold', ha='left')
text(.45, 8.22, 'A   Reference pipeline (04b)', 12, weight='bold', ha='left')
box(.5,6.8,2.4,.95,'Sensor observations','Deterministic features')
box(3.45,6.8,2.4,.95,'LSTM alarm','Predicted TTF ≤ 5,000 s')
box(6.4,6.8,2.2,.95,'RF gate','RF probability ≥ 0.5?')
box(9.45,7.3,2.45,.65,'Retained alarm',color=teal,fill='#eff8f5')
box(9.45,6.35,2.45,.65,'Discarded alarm',color=amber,fill='#fcf6eb')
arrow([(2.92,7.27),(3.43,7.27)])
arrow([(5.87,7.27),(6.38,7.27)])
arrow([(8.62,7.5),(9.0,7.5),(9.0,7.62),(9.43,7.62)])
arrow([(8.62,7.05),(9.0,7.05),(9.0,6.67),(9.43,6.67)])
text(8.98,7.85,'Yes',9); text(8.98,6.38,'No',9)
text(13.0,7.17,'RF rejection removes\nan existing LSTM alarm.',10,color=amber)
ax.plot([.45,14.55],[6.02,6.02],color='#d5dce3',lw=1)
text(.45,5.7,'B   Alarm-preserving architecture with JEV / LLM review routing',12,weight='bold',ha='left')

box(.5,3.65,2.05,1.1,'All LSTM alarms','Predicted TTF\n≤ 5,000 s')
box(3.1,3.65,1.85,1.1,'RF split','p ≥ 0.5?')
box(5.6,4.8,2.3,.65,'Tier 1','',color=teal,fill='#eff8f5')
text(8.15,5.32,'Immediate review',9,ha='left',color=teal)
box(5.6,3.65,2.3,.85,'Rescue lane','RF-rejected alarms',color=teal,fill='#eff8f5')
box(5.6,2.2,2.3,1.05,'Evidence packet','LSTM + sensor evidence\nNo RF score / labels / IDs')
box(8.5,2.2,1.8,1.05,'JEV router','Rank within\neach fault family')
box(10.9,3.65,1.95,.75,'HIGH · 25%','Priority review',color=teal,fill='#eff8f5')
box(10.9,2.2,1.95,1.05,'MID · 50%','LLM analysis\n+ numeric claim checks',color=blue,dashed=True)
box(10.9,.95,1.95,.75,'LOW · 25%','Deferred review',color=amber,fill='#fcf6eb')
box(13.5,2.2,1.0,2.2,'Human','Review\nevidence\nand decide',color=teal,fill='#eff8f5')
arrow([(2.57,4.2),(3.08,4.2)])
arrow([(4.97,4.42),(5.25,4.42),(5.25,5.12),(5.58,5.12)],teal)
text(5.17,5.43,'Yes',9)
arrow([(4.97,3.92),(5.58,3.92)])
text(5.26,3.65,'No',9)
arrow([(7.92,5.12),(14.0,5.12),(14.0,4.42)],teal)
arrow([(6.75,3.63),(6.75,3.27)])
arrow([(7.92,2.72),(8.48,2.72)])
arrow([(10.32,2.98),(10.58,2.98),(10.58,4.02),(10.88,4.02)])
arrow([(10.32,2.72),(10.88,2.72)],dashed=True)
arrow([(10.32,2.43),(10.58,2.43),(10.58,1.32),(10.88,1.32)])
arrow([(12.87,4.02),(13.48,4.02)],teal)
arrow([(12.87,2.72),(13.48,2.72)],dashed=True)
arrow([(12.87,1.32),(14.0,1.32),(14.0,2.18)],amber)
box(.5,1.15,4.45,1.55,'Every LSTM alarm remains in the review queue',
    'JEV / LLM adjust review order and analysis depth.\nUnavailable outputs retain an explicit status.\nNo automatic alarm cancellation or interlock.',color=teal,fill='#eff8f5')
text(.5,.54,'Dashed path: selective-LLM scenario. In 07b, all 271 rescue cases received LLM calls; MID-only routing (136 cases) reused saved outputs.',8.8,ha='left')
text(.5,.22,'HIGH / MID / LOW are rank allocations (25% / 50% / 25%, with integer rounding), not calibrated failure probabilities.',8.8,ha='left')

for ext in ('svg','pdf','png'):
    fig.savefig(OUT / f'S1_alarm_preserving_pipeline.{ext}', dpi=400, facecolor='white')
plt.close(fig)
print(OUT / 'S1_alarm_preserving_pipeline.png')
