"""
Heatmap 3x3 (tamano x arquitectura) para la Seccion 5.2 del TFG, uniendo
5.2.1 (efecto de tamano) y 5.2.2 (efecto de arquitectura) en una unica
figura autocontenida (sin tablas laterales).

Fuente de datos:
- Medias (color de celda): resumen_metricas_todas_condiciones.csv
  (Final_Score_mean, filtro Framework=='LangGraph'), la misma fuente que
  usa generar_figuras_tfg.py para el resto de figuras de la memoria.
- Test t pareado: recalculado aqui mismo directamente desde los CSV
  originales por debate (03_Langgraph_Parallel/data/<size>/<arch>/repN/
  judge_evaluation_results_consistent.csv), filtrando Reliable==True y
  emparejando por Q_ID+repeticion (misma metodologia que Seccion 5.1.3).
  Usa el estado ACTUAL de esos CSV (posterior a los commits 604c163d y
  8ba0f665 que corrigieron datos contaminados de 8B individual/homogeneo).

Paleta definitiva: secuencial monocromatica (Blues) -- una puntuacion
continua sin punto de referencia natural se comunica mejor con una
secuencial que con una divergente (se probaron varias alternativas:
navy a medida, naranja-verde, naranja-azul, spectral; esta fue la elegida).

Salida: figuras_tfg/heatmap_tamano_arquitectura.png
"""
import csv
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})

SIZES = ['70B', '8B', '2B']
ARCHS = ['individual', 'homogeneo', 'heterogeneo']
ARCH_ES = {'individual': 'Individual', 'homogeneo': 'Homogéneo', 'heterogeneo': 'Heterogéneo'}

def es(x, dec=2):
    return f"{x:.{dec}f}".replace('.', ',')

# ---------------------------------------------------------------
# 1. Medias (color de celda)
# ---------------------------------------------------------------
means = {}
with open('resumen_metricas_todas_condiciones.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['Framework'] != 'LangGraph':
            continue
        means[(r['Modelo'], r['Arquitectura'])] = float(r['Final_Score_mean'])

score_grid = np.array([[means[(s, a)] for a in ARCHS] for s in SIZES])

# ---------------------------------------------------------------
# 2. Test t pareado, recalculado desde los CSV crudos por debate
# ---------------------------------------------------------------
def load(size, arch):
    frames = []
    for rep in [1, 2, 3]:
        path = f'03_Langgraph_Parallel/data/{size}/{arch}/rep{rep}/judge_evaluation_results_consistent.csv'
        df = pd.read_csv(path)
        df = df[df['Reliable'] == True].copy()
        df['key'] = df['Q_ID'].astype(str) + '_rep' + str(rep)
        frames.append(df[['key', 'Final_Score']])
    return pd.concat(frames, ignore_index=True)

data = {(s, a): load(s, a) for s in SIZES for a in ARCHS}

def paired_t(c1, c2):
    d1 = data[c1].set_index('key')['Final_Score']
    d2 = data[c2].set_index('key')['Final_Score']
    common = d1.index.intersection(d2.index)
    t, p = stats.ttest_rel(d1.loc[common].values, d2.loc[common].values)
    return t, p, len(common)

arch_pairs = [('individual', 'homogeneo'), ('homogeneo', 'heterogeneo'), ('individual', 'heterogeneo')]
arch_t = {(s, a1, a2): paired_t((s, a1), (s, a2)) for s in SIZES for a1, a2 in arch_pairs}

size_pairs = [('70B', '8B'), ('8B', '2B'), ('70B', '2B')]
size_t = {(a, s1, s2): paired_t((s1, a), (s2, a)) for a in ARCHS for s1, s2 in size_pairs}

def sig(t):
    return abs(t) > 1.96

print("=== Verificacion 8B individual vs homogeneo (revisar Seccion 5.2.2 de la memoria) ===")
t, p, n = arch_t[('8B', 'individual', 'homogeneo')]
print(f"t={t:.4f}  p={p:.4f}  n={n}  -> redondeado: t={es(t)}  (la memoria debe decir esto, no t=1,93)")

print("\n=== 70B: individual -> heterogeneo (unica arquitectura significativa) ===")
t70, p70, n70 = arch_t[('70B', 'individual', 'heterogeneo')]
print(f"t={t70:.4f}  p={p70:.4f}  n={n70}  -> t={es(t70)}  sig={sig(t70)}")

# ---------------------------------------------------------------
# 3. Figura
# ---------------------------------------------------------------
CMAP = plt.get_cmap('Blues')
PAD = 0.75  # margen amplio en vmin/vmax para un degradado suave, sin saltos bruscos

fig, ax = plt.subplots(figsize=(7.3, 6.6))

vmin, vmax = score_grid.min() - PAD, score_grid.max() + PAD
im = ax.imshow(score_grid, cmap=CMAP, vmin=vmin, vmax=vmax, aspect='auto')

ax.set_xlim(-0.5, 2.5)
ax.set_xticks(range(3))
ax.set_xticklabels([ARCH_ES[a] for a in ARCHS], fontsize=11)
ax.set_yticks(range(3))
ax.set_yticklabels(SIZES, fontsize=12)
ax.set_xlabel('Arquitectura de debate', fontsize=11.5, labelpad=8)
ax.set_ylabel('Tamaño de modelo', fontsize=12, labelpad=10)
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.tick_params(length=0)

# Anotacion de puntuacion en cada celda
for i in range(3):
    for j in range(3):
        val = score_grid[i, j]
        rgba = CMAP((val - vmin) / (vmax - vmin))
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        txt_color = 'white' if luminance < 0.6 else '#102027'
        ax.text(j, i, es(val), ha='center', va='center',
                 fontsize=18, fontweight='bold', color=txt_color)

cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, shrink=0.85)
cbar.set_label('Puntuación media del juez\n(Final\\_Score, 0–10)', fontsize=9)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=2.5)
ax.tick_params(which='minor', length=0)

# --- Lineas rojas horizontales: efecto de TAMANO (Seccion 5.2.1 / Tabla 5.4) ---
adj_v = [('70B', '8B', 0, 1), ('8B', '2B', 1, 2)]
for j, a in enumerate(ARCHS):
    for s1, s2, y1, y2 in adj_v:
        t, p, n = size_t[(a, s1, s2)]
        if sig(t):
            ax.plot([j - 0.5, j + 0.5], [y1 + 0.5, y1 + 0.5],
                     color='#D81B60', lw=3.2, zorder=5)

# --- Corchete superior sobre la fila 70B: efecto de ARQUITECTURA (5.2.2) ---
# Unica comparacion de arquitectura significativa en las 9 condiciones.
y_bracket = -0.85
x_left, x_right = -0.4, 2.4
ax.set_ylim(2.5, -1.35)
ax.plot([x_left, x_right], [y_bracket, y_bracket], color='#D81B60', lw=1.6, zorder=6)
ax.plot([x_left, x_left], [y_bracket, y_bracket + 0.18], color='#D81B60', lw=1.6, zorder=6)
ax.plot([x_right, x_right], [y_bracket, y_bracket + 0.18], color='#D81B60', lw=1.6, zorder=6)
ax.text((x_left + x_right) / 2, y_bracket - 0.10, f"t={es(t70)}*",
         ha='center', va='bottom', fontsize=10.5, color='#D81B60', fontweight='bold')

# Sin titulo ni nota al pie dentro de la imagen: ambos ya los aporta el
# \caption de LaTeX en la memoria (evita duplicar el mismo texto dos veces).

out_path = 'figuras_tfg/heatmap_tamano_arquitectura.png'
fig.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"\nGuardado: {out_path}")
