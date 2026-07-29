"""
Genera resumen_metricas_todas_condiciones.csv con las estadísticas agregadas
de todas las condiciones experimentales (9 LangGraph + AutoGen + CrewAI).

Uso:
    python generar_resumen_stats.py
Salida:
    resumen_metricas_todas_condiciones.csv  (en el directorio actual)
"""

import csv
import statistics
from pathlib import Path

BASE = Path(r"c:\Users\Laura\Desktop\TFG_Laura")
OUTPUT = BASE / "resumen_metricas_todas_condiciones.csv"


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dedup(rows, key="ID"):
    seen = {}
    for r in rows:
        seen[r[key]] = r
    return list(seen.values())


def dedup_q(rows):
    return dedup(rows, key="Q_ID")


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


def process_condition(framework, modelo, arquitectura, metrics_path, eval_path):
    m = dedup(read_csv(metrics_path))
    e = dedup_q(read_csv(eval_path))
    e_rel = [r for r in e if is_reliable(r)]

    tps_col = "Avg_TPS" if "Avg_TPS" in m[0] else "TPS"
    tps_m, tps_s   = stats([r[tps_col]      for r in m])
    tot_m, tot_s   = stats([r["Total_S"]     for r in m])
    tout_m, tout_s = stats([r["Tokens_Out"]  for r in m])
    tin_m,  tin_s  = stats([r["Tokens_In"]   for r in m])

    ta_m, ta_s = stats([r["Technical_Accuracy"]  for r in e_rel])
    ca_m, ca_s = stats([r["Context_Adherence"]   for r in e_rel])
    dc_m, dc_s = stats([r["Decision_Capability"] for r in e_rel])
    fs_m, fs_s = stats([r["Final_Score"]         for r in e_rel])

    n_ev, lo, lo_pct, me, me_pct, hi, hi_pct = halluc_stats(e_rel)

    return {
        "Framework":        framework,
        "Modelo":           modelo,
        "Arquitectura":     arquitectura,
        "N_debates":        len(m),
        "N_eval_total":     n_ev,
        "N_eval_fiables":   len(e_rel),
        "TPS_mean":         tps_m,  "TPS_std":         tps_s,
        "Total_S_mean":     tot_m,  "Total_S_std":     tot_s,
        "Tokens_Out_mean":  tout_m, "Tokens_Out_std":  tout_s,
        "Tokens_In_mean":   tin_m,  "Tokens_In_std":   tin_s,
        "Tech_Acc_mean":    ta_m,   "Tech_Acc_std":    ta_s,
        "Ctx_Adh_mean":     ca_m,   "Ctx_Adh_std":     ca_s,
        "Dec_Cap_mean":     dc_m,   "Dec_Cap_std":     dc_s,
        "Final_Score_mean": fs_m,   "Final_Score_std": fs_s,
        "Halluc_Low_n":     lo,     "Halluc_Low_pct":  lo_pct,
        "Halluc_Med_n":     me,     "Halluc_Med_pct":  me_pct,
        "Halluc_High_n":    hi,     "Halluc_High_pct": hi_pct,
    }


# ── Condiciones ──────────────────────────────────────────────────────────────

conditions = []

# LangGraph: 9 condiciones (3 tamaños × 3 arquitecturas)
lg_base = BASE / "03_Langgraph_Parallel" / "data"
for size in ["2B", "8B", "70B"]:
    for arch in ["individual", "homogeneo", "heterogeneo"]:
        folder = lg_base / size / arch
        conditions.append((
            "LangGraph", size, arch,
            folder / "experiment_metrics.csv",
            folder / "judge_evaluation_results_consistent.csv",
        ))

# AutoGen
ag_base = BASE / "04_Autogen_Debate" / "Resultados"
conditions.append((
    "AutoGen", "8B", "heterogeneo",
    ag_base / "metricas_autogen.csv",
    ag_base / "judge_evaluation_results_consistent.csv",
))

# CrewAI
cr_base = BASE / "05_CrewAI_Debate" / "Resultados"
conditions.append((
    "CrewAI", "8B", "heterogeneo",
    cr_base / "metricas_crewai.csv",
    cr_base / "judge_evaluation_results_consistent.csv",
))

# ── Procesado y escritura ─────────────────────────────────────────────────────

rows = []
for framework, modelo, arch, mp, ep in conditions:
    print(f"  Procesando {framework} {modelo} {arch}...", end=" ")
    try:
        row = process_condition(framework, modelo, arch, mp, ep)
        rows.append(row)
        print(f"OK  (n={row['N_debates']}, fiables={row['N_eval_fiables']})")
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
