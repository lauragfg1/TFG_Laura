"""
Generación de figuras para el capítulo de Resultados del TFG.
Produce 4 figuras en formato PNG (300 dpi) en la carpeta ./figuras_tfg/
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# --- Estilo general ---
plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          12,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'legend.fontsize':    11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'figure.facecolor':   'white',
})

OUT = 'figuras_tfg'
os.makedirs(OUT, exist_ok=True)

# --- Colores y estilos ---
C = {'2B': '#2196F3', '8B': '#FF9800', '70B': '#2E7D32'}
LS = {'individual': '-', 'homogeneo': '--', 'heterogeneo': '-.'}
MK = {'individual': 'o', 'homogeneo': 's', 'heterogeneo': '^'}
SIZES  = ['2B', '8B', '70B']
ARCHS  = ['individual', 'homogeneo', 'heterogeneo']
ARCH_L = ['Individual', 'Homogéneo', 'Heterogéneo']

# --- Ficheros ---
BASE = '03_Langgraph_Parallel/data'
JUDGE_FILES = {
    (s, a): f'{BASE}/{s}/{a}/judge_evaluation_results_consistent.csv'
    for s in SIZES for a in ARCHS
}
PERF_FILES = {
    (s, a): f'{BASE}/{s}/{a}/experiment_metrics.csv'
    for s in SIZES for a in ARCHS
}


def load_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def reliable(rows):
    return [r for r in rows if r.get('Reliable', '').strip().lower() == 'true']


def mean(lst, key):
    vals = [float(r[key]) for r in lst if r.get(key, '') not in ('', '0', '0.0')]
    return sum(vals) / len(vals) if vals else 0


def hal_counts(rows):
    L = sum(1 for r in rows if r.get('Hallucination_Level', '') == 'Low')
    M = sum(1 for r in rows if r.get('Hallucination_Level', '') == 'Medium')
    H = sum(1 for r in rows if r.get('Hallucination_Level', '') == 'High')
    return L, M, H


# ── Precálculo de métricas ──────────────────────────────────────────────────
metrics = {}
for s in SIZES:
    for a in ARCHS:
        rows = load_csv(JUDGE_FILES[(s, a)])
        rel  = reliable(rows)
        L, M, H = hal_counts(rel)
        n = len(rel)
        metrics[(s, a)] = {
            'n':           n,
            'score':       mean(rel, 'Final_Score'),
            'acc':         mean(rel, 'Technical_Accuracy'),
            'ctx':         mean(rel, 'Context_Adherence'),
            'dec':         mean(rel, 'Decision_Capability'),
            'hal_L':       L,
            'hal_M':       M,
            'hal_H':       H,
            'hal_low_pct': 100 * L / n if n else 0,
        }

perf = {}
for s in SIZES:
    for a in ARCHS:
        rows = load_csv(PERF_FILES[(s, a)])
        totals = [float(r['Total_S']) for r in rows
                  if r.get('Total_S', '0') not in ('', '0', '0.0')]
        perf[(s, a)] = sum(totals) / len(totals) if totals else 0


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Comparativa por tamaño (modo individual)
# ══════════════════════════════════════════════════════════════════════════════
def fig1_tamano():
    fig, ax = plt.subplots(figsize=(10, 5))

    labels     = ['2B', '8B', '70B']
    score_vals = [metrics[(s, 'heterogeneo')]['score'] for s in SIZES]
    acc_vals   = [metrics[(s, 'heterogeneo')]['acc']   for s in SIZES]
    ctx_vals   = [metrics[(s, 'heterogeneo')]['ctx']   for s in SIZES]
    dec_vals   = [metrics[(s, 'heterogeneo')]['dec']   for s in SIZES]

    x      = np.arange(len(labels))
    width  = 0.19
    alpha  = 0.88
    offset = 1.5 * width   # centrar las 4 barras

    b1 = ax.bar(x - offset,          score_vals, width, label='Final Score',
                color='#1565C0', alpha=alpha, zorder=3)
    b2 = ax.bar(x - width/2,         acc_vals,   width, label='Technical Accuracy',
                color='#00897B', alpha=alpha, zorder=3)
    b3 = ax.bar(x + width/2,         ctx_vals,   width, label='Context Adherence',
                color='#F57F17', alpha=alpha, zorder=3)
    b4 = ax.bar(x + offset,          dec_vals,   width, label='Decision Capability',
                color='#8E24AA', alpha=alpha, zorder=3)

    # Etiquetas encima de cada barra
    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.04,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Tamaño del modelo')
    ax.set_ylabel('Puntuación (0–10)')
    ax.set_ylim(3.5, 8.0)
    ax.set_title('Calidad en modo heterogéneo por tamaño de modelo\n'
                 '(evaluaciones fiables)', pad=10)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    fig.tight_layout()
    path = f'{OUT}/fig_tamano_individual.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Efecto de la arquitectura sobre Final Score
# ══════════════════════════════════════════════════════════════════════════════
def fig2_arquitectura():
    fig, ax = plt.subplots(figsize=(7, 5))

    x_pos  = [0, 1, 2]
    x_labs = ['Individual', 'Homogéneo', 'Heterogéneo']

    for s in SIZES:
        y = [metrics[(s, a)]['score'] for a in ARCHS]
        ax.plot(x_pos, y, color=C[s], linewidth=2.2,
                marker='o', markersize=8, markerfacecolor='white',
                markeredgewidth=2.2, label=s, zorder=3)
        # Etiquetas de valor
        for xi, yi in zip(x_pos, y):
            ax.annotate(f'{yi:.2f}',
                        xy=(xi, yi),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', fontsize=9.5, color=C[s], fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labs)
    ax.set_ylabel('Final Score (evaluaciones fiables)')
    ax.set_xlabel('Arquitectura de debate')
    ax.set_ylim(3.6, 6.4)
    ax.set_title('Efecto de la arquitectura sobre la calidad de respuesta\npor familia de modelo', pad=10)
    ax.legend(title='Modelo', framealpha=0.9)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

    # Línea de referencia: 5.0
    ax.axhline(5.0, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.text(2.08, 5.02, '5.0', color='gray', fontsize=9, va='bottom')

    fig.tight_layout()
    path = f'{OUT}/fig_arquitectura_score.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 3 — Scatter coste-calidad
# ══════════════════════════════════════════════════════════════════════════════
def fig3_coste_calidad():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    OFFSETS = {
        ('70B','individual'):  ( 20,  8),
        ('70B','homogeneo'):   ( 20,  8),
        ('70B','heterogeneo'): ( 20, -14),
        ('8B', 'individual'):  (-80, -15),
        ('8B', 'homogeneo'):   ( 10, -15),
        ('8B', 'heterogeneo'): ( 10,  8),
        ('2B', 'individual'):  (-75,  8),
        ('2B', 'homogeneo'):   ( 10,  8),
        ('2B', 'heterogeneo'): ( 10, -14),
    }
    ARCH_SYM = {'individual': 'o', 'homogeneo': 's', 'heterogeneo': '^'}
    ARCH_NAME = {'individual': 'Ind.', 'homogeneo': 'Homo.', 'heterogeneo': 'Heter.'}

    for s in SIZES:
        for a in ARCHS:
            t  = perf[(s, a)]
            sc = metrics[(s, a)]['score']
            dx, dy = OFFSETS[(s, a)]
            ax.scatter(t, sc, color=C[s], marker=ARCH_SYM[a],
                       s=100, zorder=5, edgecolors='white', linewidths=0.8)
            label = f'{s}-{ARCH_NAME[a]}'
            ax.annotate(label, xy=(t, sc),
                        xytext=(dx, dy), textcoords='offset points',
                        fontsize=9, color=C[s], fontweight='bold',
                        arrowprops=dict(arrowstyle='-', color=C[s],
                                        lw=0.7, alpha=0.6))

    ax.set_xscale('log')
    ax.set_xlabel('Tiempo medio por debate (s) — escala log')
    ax.set_ylabel('Final Score (evaluaciones fiables)')
    ax.set_title('Frontera coste-calidad por condición experimental', pad=10)
    ax.set_ylim(3.8, 6.2)

    # Leyendas manuales
    size_handles = [plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=C[s], markersize=9, label=s)
                    for s in SIZES]
    arch_handles = [plt.Line2D([0], [0], marker=ARCH_SYM[a], color='gray',
                               markersize=8, linestyle='None',
                               label=ARCH_NAME[a])
                    for a in ARCHS]

    leg1 = ax.legend(handles=size_handles, title='Tamaño',
                     loc='upper left', framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=arch_handles, title='Arq.',
              loc='lower right', framealpha=0.9)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{int(x):,}s'))

    fig.tight_layout()
    path = f'{OUT}/fig_coste_calidad.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 4 — Alucinaciones (barras apiladas % + línea Hal.Low%)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_alucinaciones():
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    # Etiquetas del eje X
    conds  = [(s, a) for s in SIZES for a in ARCHS]
    x_labs = [f'{s}\n{ARCH_NAME}' for s in SIZES for a, ARCH_NAME in
              zip(ARCHS, ['Ind.', 'Homo.', 'Heter.'])]
    x_pos  = np.arange(len(conds))
    width  = 0.6

    # Porcentajes
    pct_L, pct_M, pct_H = [], [], []
    hl_pct = []
    for (s, a) in conds:
        m  = metrics[(s, a)]
        n  = m['n']
        pct_L.append(100 * m['hal_L'] / n if n else 0)
        pct_M.append(100 * m['hal_M'] / n if n else 0)
        pct_H.append(100 * m['hal_H'] / n if n else 0)
        hl_pct.append(m['hal_low_pct'])

    C_LOW  = '#43A047'   # verde
    C_MED  = '#FFA726'   # naranja
    C_HIGH = '#E53935'   # rojo

    b_L = ax1.bar(x_pos, pct_L, width, label='Low',    color=C_LOW,  alpha=0.88, zorder=3)
    b_M = ax1.bar(x_pos, pct_M, width, bottom=pct_L,
                  label='Medium', color=C_MED,  alpha=0.88, zorder=3)
    b_H = ax1.bar(x_pos, pct_H, width,
                  bottom=[l + m for l, m in zip(pct_L, pct_M)],
                  label='High',   color=C_HIGH, alpha=0.88, zorder=3)

    # Línea Hal. Low %
    ax2 = ax1.twinx()
    ax2.plot(x_pos, hl_pct, color='#1A237E', linewidth=2,
             marker='D', markersize=7, markerfacecolor='white',
             markeredgewidth=2, label='Hal. Low %', zorder=6)
    ax2.set_ylabel('Hal. Low % (evaluaciones fiables)', color='#1A237E')
    ax2.tick_params(axis='y', labelcolor='#1A237E')
    ax2.set_ylim(0, 110)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}%'))

    # Etiquetas del valor Low% sobre cada punto
    for xi, yp in zip(x_pos, hl_pct):
        ax2.annotate(f'{yp:.0f}%', xy=(xi, yp),
                     xytext=(0, 9), textcoords='offset points',
                     ha='center', fontsize=8.5, color='#1A237E', fontweight='bold')

    # Separadores por grupo de tamaño
    for sep in [2.5, 5.5]:
        ax1.axvline(sep, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

    # Etiquetas de grupo
    for xi, label in [(1, '2B'), (4, '8B'), (7, '70B')]:
        ax1.text(xi, 103, label, ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color='#333333')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labs, fontsize=10)
    ax1.set_ylabel('Distribución de alucinaciones (%)')
    ax1.set_ylim(0, 115)
    ax1.set_title('Distribución de niveles de alucinación por condición experimental\n'
                  '(barras: % Low/Medium/High; línea: tasa Low en evaluaciones fiables)', pad=10)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}%'))

    # Leyendas
    handles_bar = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.88, label=l)
                   for c, l in [(C_LOW,'Low'), (C_MED,'Medium'), (C_HIGH,'High')]]
    handle_line = plt.Line2D([0], [0], color='#1A237E', linewidth=2,
                             marker='D', markersize=7, markerfacecolor='white',
                             markeredgewidth=2, label='Hal. Low %')
    ax1.legend(handles=handles_bar + [handle_line],
               loc='lower left', framealpha=0.9, ncol=2)

    fig.tight_layout()
    path = f'{OUT}/fig_alucinaciones.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 5 — 70B: comparativa de las 4 métricas por arquitectura
# ══════════════════════════════════════════════════════════════════════════════
def fig5_70b_arquitecturas():
    fig, ax = plt.subplots(figsize=(10, 5))

    arch_labels = ['Individual', 'Homogéneo', 'Heterogéneo']
    score_vals  = [metrics[('70B', a)]['score'] for a in ARCHS]
    acc_vals    = [metrics[('70B', a)]['acc']   for a in ARCHS]
    ctx_vals    = [metrics[('70B', a)]['ctx']   for a in ARCHS]
    dec_vals    = [metrics[('70B', a)]['dec']   for a in ARCHS]

    x      = np.arange(len(arch_labels))
    width  = 0.19
    offset = 1.5 * width
    alpha  = 0.88

    b1 = ax.bar(x - offset,  score_vals, width, label='Final Score',
                color='#1565C0', alpha=alpha, zorder=3)
    b2 = ax.bar(x - width/2, acc_vals,   width, label='Technical Accuracy',
                color='#00897B', alpha=alpha, zorder=3)
    b3 = ax.bar(x + width/2, ctx_vals,   width, label='Context Adherence',
                color='#F57F17', alpha=alpha, zorder=3)
    b4 = ax.bar(x + offset,  dec_vals,   width, label='Decision Capability',
                color='#8E24AA', alpha=alpha, zorder=3)

    # Etiquetas de valor
    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.04,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    # Flecha de mejora Final Score: barra individual → heterogéneo
    # Se coloca por encima de todos los labels de barras (~7.1 máx)
    y_ind   = score_vals[0]
    y_het   = score_vals[2]
    delta   = y_het - y_ind
    y_arrow = 7.7   # por encima del label más alto
    x_ind   = x[0] - offset
    x_het   = x[2] - offset
    x_mid   = (x_ind + x_het) / 2

    # Líneas verticales punteadas desde tope de barra hasta la flecha
    ax.plot([x_ind, x_ind], [y_ind + 0.08, y_arrow - 0.05],
            color='#1565C0', lw=1.0, ls=':', alpha=0.6)
    ax.plot([x_het, x_het], [y_het + 0.08, y_arrow - 0.05],
            color='#1565C0', lw=1.0, ls=':', alpha=0.6)

    # Flecha horizontal doble
    ax.annotate('', xy=(x_het, y_arrow), xytext=(x_ind, y_arrow),
                arrowprops=dict(arrowstyle='<->', color='#1565C0', lw=1.8))

    # Texto encima de la flecha
    ax.text(x_mid, y_arrow + 0.13,
            f'+{delta:.2f} ({100 * delta / y_ind:.1f}%)',
            ha='center', va='bottom', fontsize=10,
            color='#1565C0', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels)
    ax.set_xlabel('Arquitectura de debate')
    ax.set_ylabel('Puntuación (0–10)')
    ax.set_ylim(3.5, 8.5)
    ax.set_title('Modelo 70B — evolución de métricas de calidad por arquitectura\n'
                 '(evaluaciones fiables)', pad=10)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    fig.tight_layout()
    path = f'{OUT}/fig_70b_arquitecturas.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


# ── Ejecución ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generando figuras...')
    fig1_tamano()
    fig2_arquitectura()
    fig3_coste_calidad()
    fig4_alucinaciones()
    fig5_70b_arquitecturas()
    print(f'\nFiguras guardadas en: {os.path.abspath(OUT)}/')
