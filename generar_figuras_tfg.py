"""
Generacion de figuras para el capitulo de Resultados del TFG.
Produce las figuras en formato PNG (300 dpi) en la carpeta ./figuras_tfg/

Lee resumen_metricas_todas_condiciones.csv (generado por
generar_resumen_stats.py, que agrega las 3 repeticiones de cada condicion)
en vez de recalcular medias/desviaciones aqui: asi las figuras y las tablas
de la memoria salen siempre del mismo numero.

Uso:
    python generar_resumen_stats.py   (primero, si los datos han cambiado)
    python generar_figuras_tfg.py
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

RESUMEN_CSV = 'resumen_metricas_todas_condiciones.csv'

# --- Colores y estilos ---
C = {'2B': '#2196F3', '8B': '#FF9800', '70B': '#2E7D32'}
SIZES  = ['2B', '8B', '70B']
ARCHS  = ['individual', 'homogeneo', 'heterogeneo']
ARCH_ES = {'individual': 'Individual', 'homogeneo': 'Homogéneo', 'heterogeneo': 'Heterogéneo'}
ARCH_COLOR = {'individual': '#78909C', 'homogeneo': '#5C6BC0', 'heterogeneo': '#EF5350'}

# Nombres de metricas traducidos, para que ninguna figura mezcle ingles/espanol
CRIT_ES = {
    'score': 'Puntuación\nFinal',
    'acc':   'Precisión\nTécnica',
    'ctx':   'Adherencia\nal Contexto',
    'dec':   'Capacidad\nde Decisión',
}
CRIT_ORDER = ['score', 'acc', 'ctx', 'dec']


def load_resumen():
    with open(RESUMEN_CSV, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    metrics, perf = {}, {}
    for r in rows:
        if r['Framework'] != 'LangGraph':
            continue
        key = (r['Modelo'], r['Arquitectura'])
        n = int(r['N_eval_fiables'])
        metrics[key] = {
            'n':           n,
            'score':       float(r['Final_Score_mean']),
            'acc':         float(r['Tech_Acc_mean']),
            'ctx':         float(r['Ctx_Adh_mean']),
            'dec':         float(r['Dec_Cap_mean']),
            'hal_L':       int(r['Halluc_Low_n']),
            'hal_M':       int(r['Halluc_Med_n']),
            'hal_H':       int(r['Halluc_High_n']),
            'hal_low_pct': float(r['Halluc_Low_pct']),
        }
        perf[key] = float(r['Total_S_mean'])
    return metrics, perf


metrics, perf = load_resumen()


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Comparativa de calidad por tamaño (modo heterogéneo)
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

    b1 = ax.bar(x - offset,          score_vals, width, label='Puntuación Final',
                color='#1565C0', alpha=alpha, zorder=3)
    b2 = ax.bar(x - width/2,         acc_vals,   width, label='Precisión Técnica',
                color='#00897B', alpha=alpha, zorder=3)
    b3 = ax.bar(x + width/2,         ctx_vals,   width, label='Adherencia al Contexto',
                color='#F57F17', alpha=alpha, zorder=3)
    b4 = ax.bar(x + offset,          dec_vals,   width, label='Capacidad de Decisión',
                color='#8E24AA', alpha=alpha, zorder=3)

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.04,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('Tamaño del modelo')
    ax.set_ylabel('Puntuación (0–10)')
    ax.set_ylim(0, max(score_vals + acc_vals + ctx_vals + dec_vals) + 1.2)
    ax.set_title('Calidad en modo heterogéneo por tamaño de modelo\n'
                 '(evaluaciones fiables, media de 3 repeticiones)', pad=10)
    ax.legend(loc='lower right', framealpha=0.9, ncol=2)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    fig.tight_layout()
    path = f'{OUT}/fig_calidad_por_tamano_heterogeneo.png'
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
        for xi, yi in zip(x_pos, y):
            ax.annotate(f'{yi:.2f}',
                        xy=(xi, yi),
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', fontsize=9.5, color=C[s], fontweight='bold')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labs)
    ax.set_ylabel('Puntuación Final (evaluaciones fiables)')
    ax.set_xlabel('Arquitectura de debate')
    y_all = [metrics[(s, a)]['score'] for s in SIZES for a in ARCHS]
    ax.set_ylim(min(y_all) - 0.6, max(y_all) + 0.6)
    ax.set_title('Efecto de la arquitectura sobre la calidad de respuesta\npor tamaño de modelo', pad=10)
    ax.legend(title='Modelo', framealpha=0.9)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))

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

    ARCH_SYM = {'individual': 'o', 'homogeneo': 's', 'heterogeneo': '^'}
    ARCH_NAME = {'individual': 'Ind.', 'homogeneo': 'Homo.', 'heterogeneo': 'Heter.'}

    for s in SIZES:
        for a in ARCHS:
            t  = perf[(s, a)]
            sc = metrics[(s, a)]['score']
            ax.scatter(t, sc, color=C[s], marker=ARCH_SYM[a],
                       s=100, zorder=5, edgecolors='white', linewidths=0.8)
            label = f'{s}-{ARCH_NAME[a]}'
            ax.annotate(label, xy=(t, sc),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=8.5, color=C[s], fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlabel('Tiempo medio por debate (s) — escala log')
    ax.set_ylabel('Puntuación Final (evaluaciones fiables)')
    ax.set_title('Frontera coste-calidad por condición experimental\n(media de 3 repeticiones)', pad=10)
    y_all = [metrics[(s, a)]['score'] for s in SIZES for a in ARCHS]
    ax.set_ylim(min(y_all) - 0.6, max(y_all) + 0.6)

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
# FIGURA 4 — Alucinaciones (barras apiladas %, sin línea de progresión)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_alucinaciones():
    fig, ax = plt.subplots(figsize=(11, 5.5))

    conds  = [(s, a) for s in SIZES for a in ARCHS]
    x_labs = [f'{s}\n{ARCH_ES[a][:5]}.' for s, a in conds]
    x_pos  = np.arange(len(conds))
    width  = 0.6

    pct_L, pct_M, pct_H = [], [], []
    for (s, a) in conds:
        m = metrics[(s, a)]
        n = m['n']
        pct_L.append(100 * m['hal_L'] / n if n else 0)
        pct_M.append(100 * m['hal_M'] / n if n else 0)
        pct_H.append(100 * m['hal_H'] / n if n else 0)

    C_LOW, C_MED, C_HIGH = '#43A047', '#FFA726', '#E53935'

    b_L = ax.bar(x_pos, pct_L, width, label='Bajo',  color=C_LOW,  alpha=0.88, zorder=3)
    b_M = ax.bar(x_pos, pct_M, width, bottom=pct_L,
                 label='Medio', color=C_MED, alpha=0.88, zorder=3)
    b_H = ax.bar(x_pos, pct_H, width,
                 bottom=[l + m for l, m in zip(pct_L, pct_M)],
                 label='Alto', color=C_HIGH, alpha=0.88, zorder=3)

    # Porcentaje "Bajo" como etiqueta de texto estatica sobre cada barra
    # (sin linea de union: no representa una progresion entre condiciones,
    # cada barra es una condicion experimental independiente).
    for xi, yl in zip(x_pos, pct_L):
        ax.annotate(f'{yl:.0f}%', xy=(xi, yl),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=8.5, color='#1B5E20', fontweight='bold')

    for sep in [2.5, 5.5]:
        ax.axvline(sep, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)

    for xi, label in [(1, '2B'), (4, '8B'), (7, '70B')]:
        ax.text(xi, 103, label, ha='center', va='bottom',
                fontsize=12, fontweight='bold', color='#333333')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labs, fontsize=10)
    ax.set_ylabel('Distribución de alucinaciones (%)')
    ax.set_ylim(0, 112)
    ax.set_title('Distribución de niveles de alucinación por condición experimental\n'
                 '(% Bajo/Medio/Alto sobre evaluaciones fiables, media de 3 repeticiones)', pad=10)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}%'))
    ax.legend(loc='lower left', framealpha=0.9, ncol=3)

    fig.tight_layout()
    path = f'{OUT}/fig_alucinaciones.png'
    fig.savefig(path)
    plt.close(fig)
    print(f'  ✓ {path}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 5 (x3) — Evolución por arquitectura, una figura por tamaño de modelo.
# Eje X = criterio evaluado; leyenda = arquitectura (3 barras por criterio).
# ══════════════════════════════════════════════════════════════════════════════
def fig5_arquitectura_por_criterio(size):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    x      = np.arange(len(CRIT_ORDER))
    width  = 0.26
    alpha  = 0.88

    bars_por_arch = {}
    for i, arch in enumerate(ARCHS):
        vals = [metrics[(size, arch)][crit] for crit in CRIT_ORDER]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, label=ARCH_ES[arch],
                      color=ARCH_COLOR[arch], alpha=alpha, zorder=3)
        bars_por_arch[arch] = bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8.5)

    # Flecha de mejora en Puntuacion Final: Individual -> Heterogeneo
    y_ind = metrics[(size, 'individual')]['score']
    y_het = metrics[(size, 'heterogeneo')]['score']
    delta = y_het - y_ind
    y_max = max(metrics[(size, a)][c] for a in ARCHS for c in CRIT_ORDER)
    y_arrow = y_max + 0.9
    x_ind = 0 + (0 - 1) * width
    x_het = 0 + (2 - 1) * width

    ax.plot([x_ind, x_ind], [y_ind + 0.08, y_arrow - 0.05], color='#37474F', lw=1.0, ls=':', alpha=0.6)
    ax.plot([x_het, x_het], [y_het + 0.08, y_arrow - 0.05], color='#37474F', lw=1.0, ls=':', alpha=0.6)
    ax.annotate('', xy=(x_het, y_arrow), xytext=(x_ind, y_arrow),
                arrowprops=dict(arrowstyle='<->', color='#37474F', lw=1.6))
    sign = '+' if delta >= 0 else ''
    pct = 100 * delta / y_ind if y_ind else 0
    ax.text((x_ind + x_het) / 2, y_arrow + 0.12,
            f'{sign}{delta:.2f} ({sign}{pct:.1f}%)',
            ha='center', va='bottom', fontsize=9.5, color='#37474F', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([CRIT_ES[c] for c in CRIT_ORDER])
    ax.set_xlabel('Criterio evaluado por el juez')
    ax.set_ylabel('Puntuación (0–10)')
    ax.set_ylim(0, y_arrow + 0.8)
    ax.set_title(f'Modelo {size} — puntuación por criterio y arquitectura de debate\n'
                 '(evaluaciones fiables, media de 3 repeticiones)', pad=10)
    ax.legend(title='Arquitectura', loc='upper right', framealpha=0.9)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))

    fig.tight_layout()
    path = f'{OUT}/fig_arquitecturas_{size}.png'
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
    for _size in SIZES:
        fig5_arquitectura_por_criterio(_size)
    print(f'\nFiguras guardadas en: {os.path.abspath(OUT)}/')
