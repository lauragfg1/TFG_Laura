"""
Estima cuanto tiempo del debate 70B heterogeneo es solo recarga de modelos
en VRAM, y que tiempo hipotetico se habria tenido si los 3 modelos (el
moderador/sintetizador y los dos expertos, cada uno de una familia
distinta) hubieran cabido a la vez en memoria.

Contexto (ver 01_Langgraph_Debate/ollama_client.py y nodes.py): en la
condicion heterogenea el cliente usa keep_alive=0 porque los 3 modelos de
70B no caben simultaneamente en los 72 GB de VRAM disponibles, asi que cada
llamada expulsa el modelo anterior y recarga el que toca. Cada llamada deja
su tiempo de carga real en metrics_XXX.json (campo "tiempo_carga_s" por
llamada, con el nombre de modelo en "modelo").

Metodologia de la estimacion (por debate):
  - Coste de carga REAL de un modelo = el que ya se mide siempre en su
    primera aparicion dentro del debate (ese coste es inevitable exista o
    no suficiente VRAM: el modelo tiene que entrar en memoria alguna vez).
  - Coste de carga HIPOTETICO en las apariciones siguientes de ese mismo
    modelo dentro del mismo debate = 0, porque con VRAM suficiente para
    tener los 3 modelos residentes, esa llamada no necesitaria recargar
    nada.
  - tiempo_hipotetico_total = tiempo_real_total - (carga_real_total - carga_hipotetica_total)

Uso:
    python estimar_recarga_70b.py
Salida:
    estimacion_recarga_70b_heterogeneo.csv (detalle por debate)
    Resumen impreso por pantalla, listo para pegar en la memoria (§5.3.4).
"""

import csv
import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent
COND_DIR = BASE / "01_Langgraph_Debate" / "data" / "70B" / "heterogeneo"
RESUMEN_CSV = BASE / "resumen_metricas_todas_condiciones.csv"
OUTPUT = BASE / "estimacion_recarga_70b_heterogeneo.csv"
REPS = (1, 2, 3)


def analizar_debate(metrics_path: Path):
    with open(metrics_path, encoding="utf-8") as f:
        calls = json.load(f)
    if not calls:
        return None

    seen_models = set()
    actual_total = actual_load = hyp_load = 0.0
    load_por_modelo_real = {}   # solo la 1a carga de cada modelo, por debate

    for call in calls:
        modelo = call.get("modelo", "unknown")
        t_total = float(call.get("tiempo_total_s", 0.0) or 0.0)
        t_load = float(call.get("tiempo_carga_s", 0.0) or 0.0)

        actual_total += t_total
        actual_load += t_load

        if modelo not in seen_models:
            hyp_load += t_load
            load_por_modelo_real[modelo] = t_load
            seen_models.add(modelo)

    ahorro = actual_load - hyp_load
    hyp_total = actual_total - ahorro

    return {
        "n_calls": len(calls),
        "n_modelos_distintos": len(seen_models),
        "actual_total_s": actual_total,
        "actual_load_s": actual_load,
        "hyp_load_s": hyp_load,
        "ahorro_s": ahorro,
        "hyp_total_s": hyp_total,
        "load_por_modelo_real": load_por_modelo_real,
    }


def main():
    filas = []
    load_primera_carga = {}  # modelo -> lista de tiempos de 1a carga (para media por modelo)

    for rep in REPS:
        rep_dir = COND_DIR / f"rep{rep}"
        if not rep_dir.exists():
            continue
        for q_dir in sorted(rep_dir.glob("q*")):
            for metrics_file in sorted(q_dir.glob("metrics_*.json")):
                r = analizar_debate(metrics_file)
                if not r:
                    continue
                for modelo, t in r["load_por_modelo_real"].items():
                    load_primera_carga.setdefault(modelo, []).append(t)
                filas.append({
                    "rep": rep,
                    "q": q_dir.name,
                    "n_calls": r["n_calls"],
                    "n_modelos_distintos": r["n_modelos_distintos"],
                    "actual_total_s": round(r["actual_total_s"], 3),
                    "actual_load_s": round(r["actual_load_s"], 3),
                    "hyp_load_s": round(r["hyp_load_s"], 3),
                    "ahorro_s": round(r["ahorro_s"], 3),
                    "hyp_total_s": round(r["hyp_total_s"], 3),
                })

    if not filas:
        print(f"⚠️  No se encontraron metrics_*.json bajo {COND_DIR}")
        return

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        writer.writeheader()
        writer.writerows(filas)

    actual_totals = [r["actual_total_s"] for r in filas]
    hyp_totals = [r["hyp_total_s"] for r in filas]
    ahorros = [r["ahorro_s"] for r in filas]
    pct_ahorros = [100 * a / t for a, t in zip(ahorros, actual_totals) if t]

    print(f"Debates analizados: {len(filas)} (bajo {COND_DIR.relative_to(BASE)}, reps {REPS})\n")

    print("=== Tiempo REAL por debate (con recarga en cada llamada) ===")
    print(f"  media = {statistics.mean(actual_totals):.1f} s  "
          f"(desv. = {statistics.stdev(actual_totals):.1f} s)")

    print("\n=== Tiempo HIPOTÉTICO por debate (los 3 modelos caben a la vez en VRAM) ===")
    print(f"  media = {statistics.mean(hyp_totals):.1f} s  "
          f"(desv. = {statistics.stdev(hyp_totals):.1f} s)")

    print("\n=== Ahorro estimado ===")
    print(f"  media = {statistics.mean(ahorros):.1f} s por debate "
          f"({statistics.mean(pct_ahorros):.1f}% del tiempo total)")
    print(f"  equivalente a pasar de "
          f"{statistics.mean(actual_totals)/60:.1f} min a "
          f"{statistics.mean(hyp_totals)/60:.1f} min por debate")

    print("\n=== Coste medio de la PRIMERA carga, por modelo (inevitable aunque sobre VRAM) ===")
    for modelo, tiempos in sorted(load_primera_carga.items()):
        print(f"  {modelo}: {statistics.mean(tiempos):.1f} s (n={len(tiempos)})")

    # Contraste con individual/homogeneo (keep_alive=-1, sin recarga forzada
    # tras la primera llamada) usando el resumen ya calculado, solo como
    # referencia narrativa -- no repite el calculo, solo lo imprime si existe.
    if RESUMEN_CSV.exists():
        with open(RESUMEN_CSV, encoding="utf-8") as f:
            resumen = {(r["Framework"], r["Modelo"], r["Arquitectura"]): r
                       for r in csv.DictReader(f)}
        print("\n=== Referencia: tiempo medio real ya reportado en otras arquitecturas 70B ===")
        for arch in ["individual", "homogeneo", "heterogeneo"]:
            row = resumen.get(("LangGraph", "70B", arch))
            if row:
                print(f"  70B {arch}: {row['Total_S_mean']} s (media de 3 reps)")

    print(f"\nDetalle por debate guardado en: {OUTPUT}")


if __name__ == "__main__":
    main()
