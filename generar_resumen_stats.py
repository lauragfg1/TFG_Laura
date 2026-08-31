"""
Genera resumen_metricas_todas_condiciones.csv con las estadisticas agregadas
de todas las condiciones experimentales (9 LangGraph + AutoGen + CrewAI).

A partir del fix de acumulacion de contexto en LangGraph (RECENT_WINDOW en
nodes.py) cada condicion se ejecuto 3 veces (rep1/rep2/rep3, 50 preguntas
cada una). Este script agrupa las 3 repeticiones de cada condicion en una
sola muestra combinada (pooling) antes de calcular media/desviacion, en
lugar de promediar solo una repeticion como en la version anterior.

Es la UNICA fuente de verdad para las tablas/figuras del capitulo de
Evaluacion Experimental: generar_figuras_tfg.py y stats_comparativa.py leen
su salida en vez de repetir esta logica de agregacion.

Uso:
    python generar_resumen_stats.py
Salida:
    resumen_metricas_todas_condiciones.csv  (en el directorio actual)
"""

import csv
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "resumen_metricas_todas_condiciones.csv"

REPS = (1, 2, 3)


def read_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dedup(rows, key):
    """Si una repeticion se relanzo parcialmente puede haber filas duplicadas
    para el mismo ID/Q_ID (se re-evaluo una pregunta ya evaluada); nos
    quedamos con la ultima ocurrencia de cada clave."""
    seen = {}
    for r in rows:
        seen[r[key]] = r
    return list(seen.values())


def stats(vals):
    v = [float(x) for x in vals if x not in ("", None, "0", "0.0") and float(x) != 0.0]
    if not v:
        return 0.0, 0.0
    mean = statistics.mean(v)
    std = statistics.stdev(v) if len(v) > 1 else 0.0
    return round(mean, 2), round(std, 2)


def is_reliable(row):
    return row.get("Reliable", "").strip() == "True"


def halluc_stats(rows):
    levels = [r["Hallucination_Level"].strip() for r in rows]
    n = len(levels)
    lo = levels.count("Low")
    me = levels.count("Medium")
    hi = levels.count("High")
    pct = lambda k: round(100 * k / n, 1) if n else 0.0
    return n, lo, pct(lo), me, pct(me), hi, pct(hi)


def load_condition(base_dir: Path, metrics_filename: str,
                    metrics_key="ID", eval_key="Q_ID", reps=REPS):
    """Recorre base_dir/rep1, rep2, rep3 (los que existan), deduplica cada
    repeticion por separado (Q_ID/ID se reinicia en cada repeticion, asi
    que NO se puede deduplicar despues de concatenar) y devuelve las listas
    ya combinadas (pooled) junto con las repeticiones realmente encontradas.
    """
    pooled_metrics, pooled_eval = [], []
    reps_found = []

    for rep in reps:
        rep_dir = base_dir / f"rep{rep}"
        m_path = rep_dir / metrics_filename
        e_path = rep_dir / "judge_evaluation_results_consistent.csv"

        if not m_path.exists():
            continue
        reps_found.append(rep)

        pooled_metrics.extend(dedup(read_csv(m_path), key=metrics_key))
        if e_path.exists():
            pooled_eval.extend(dedup(read_csv(e_path), key=eval_key))

    return pooled_metrics, pooled_eval, reps_found


def process_condition(framework, modelo, arquitectura, base_dir: Path,
                       metrics_filename: str):
    m, e, reps_found = load_condition(base_dir, metrics_filename)
    if not m:
        raise FileNotFoundError(f"Sin datos en {base_dir} (reps buscadas: {REPS})")

    e_rel = [r for r in e if is_reliable(r)]

    # AutoGen/CrewAI no exponen Load_S/Prompt_S/Gen_S reales (endpoint /v1,
    # ver ollama_client.py); esas columnas quedan a 0.0 y `stats()` ya las
    # filtra como si faltaran.
    tps_col = "Avg_TPS" if "Avg_TPS" in m[0] else "TPS"
    tps_m, tps_s = stats([r[tps_col] for r in m])
    tot_m, tot_s = stats([r["Total_S"] for r in m])
    tout_m, tout_s = stats([r["Tokens_Out"] for r in m])
    tin_m, tin_s = stats([r["Tokens_In"] for r in m])

    ta_m, ta_s = stats([r["Technical_Accuracy"] for r in e_rel])
    ca_m, ca_s = stats([r["Context_Adherence"] for r in e_rel])
    dc_m, dc_s = stats([r["Decision_Capability"] for r in e_rel])
    fs_m, fs_s = stats([r["Final_Score"] for r in e_rel])

    n_ev, lo, lo_pct, me, me_pct, hi, hi_pct = halluc_stats(e_rel)

    return {
        "Framework": framework,
        "Modelo": modelo,
        "Arquitectura": arquitectura,
        "Reps_usadas": "+".join(str(r) for r in reps_found),
        "N_debates": len(m),
        "N_eval_total": len(e),
        "N_eval_fiables": len(e_rel),
        "TPS_mean": tps_m, "TPS_std": tps_s,
        "Total_S_mean": tot_m, "Total_S_std": tot_s,
        "Tokens_Out_mean": tout_m, "Tokens_Out_std": tout_s,
        "Tokens_In_mean": tin_m, "Tokens_In_std": tin_s,
        "Tech_Acc_mean": ta_m, "Tech_Acc_std": ta_s,
        "Ctx_Adh_mean": ca_m, "Ctx_Adh_std": ca_s,
        "Dec_Cap_mean": dc_m, "Dec_Cap_std": dc_s,
        "Final_Score_mean": fs_m, "Final_Score_std": fs_s,
        "Halluc_Low_n": lo, "Halluc_Low_pct": lo_pct,
        "Halluc_Med_n": me, "Halluc_Med_pct": me_pct,
        "Halluc_High_n": hi, "Halluc_High_pct": hi_pct,
    }


# -- Condiciones --------------------------------------------------------------

conditions = []

# LangGraph: 9 condiciones (3 tamanos x 3 arquitecturas), 3 reps cada una
lg_base = BASE / "03_Langgraph_Parallel" / "data"
for size in ["2B", "8B", "70B"]:
    for arch in ["individual", "homogeneo", "heterogeneo"]:
        conditions.append((
            "LangGraph", size, arch,
            lg_base / size / arch,
            "experiment_metrics.csv",
        ))

# AutoGen: condicion principal 8B heterogeneo, 3 reps
conditions.append((
    "AutoGen", "8B", "heterogeneo",
    BASE / "04_Autogen_Debate" / "Resultados",
    "metricas_autogen.csv",
))

# CrewAI: condicion principal 8B heterogeneo, 3 reps
conditions.append((
    "CrewAI", "8B", "heterogeneo",
    BASE / "05_CrewAI_Debate" / "Resultados",
    "metricas_crewai.csv",
))

# NOTA: existe un experimento adicional en curso/incompleto
# (05_CrewAI_Debate/Resultados/2B, 04_Autogen_Debate/Resultados/2B) que
# compararia AutoGen/CrewAI tambien a escala 2B. No se incluye aqui hasta
# que este completo y decidido si entra en la memoria (ver run_2B_crewai_autogen.ps1).

# -- Procesado y escritura -----------------------------------------------------

rows = []
for framework, modelo, arch, base_dir, metrics_filename in conditions:
    print(f"  Procesando {framework} {modelo} {arch}...", end=" ")
    try:
        row = process_condition(framework, modelo, arch, base_dir, metrics_filename)
        rows.append(row)
        print(f"OK  (reps={row['Reps_usadas']}, n={row['N_debates']}, fiables={row['N_eval_fiables']})")
    except Exception as exc:
        print(f"ERROR: {exc}")

if rows:
    fieldnames = list(rows[0].keys())
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nGuardado en: {OUTPUT}")
    print(f"Total condiciones: {len(rows)}")
