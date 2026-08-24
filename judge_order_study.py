#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estudio de sensibilidad al orden del prompt del juez.

Responde al comentario del tutor: "no tengo claro que el orden invertido sea
adecuado, quizás habría que explorar otras posibilidades". En vez de probar
solo 2 órdenes (el enfoque anterior), evalúa cada debate ya generado con las
4 variantes definidas en ORDER_VARIANTS (judge_evaluator_robust.py) y mide
cuánto varía la nota final según el orden — la variante más estable (menor
rango entre sus 4 notas, en promedio) es la más defendible para usar en la
campaña completa, porque su juicio depende menos de dónde se coloca cada
bloque de texto y más del contenido real.

No genera debates nuevos: reutiliza los decision_*.json/*_decision.json ya
guardados bajo 03_Langgraph_Parallel/data/.

Uso:
    python judge_order_study.py
Salida:
    judge_order_study_results.csv
"""

import csv
import json
import statistics
import time
from pathlib import Path

from judge_evaluator_robust import ORDER_VARIANTS, evaluate_once

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "03_Langgraph_Parallel" / "data"
OUTPUT = BASE / "judge_order_study_results.csv"

# N casos por condición (tamaño x modo). 10 = todos los generados en el lote
# de validación con el pipeline ya arreglado (10 preguntas x 9 condiciones).
# El estudio original (n=5/condición) se hizo sobre contexto casi puramente
# narrativo, antes del fix del RAG que garantiza contenido "structured"
# (tablas/cifras) en la recuperación — ver judge_order_study_results_pre_rag_fix.csv.
SAMPLE_PER_CONDITION = 10


def find_decision_files():
    """Recorre data/<size>/<mode>/rep*/q*/ buscando decision_*.json.

    Devuelve como mucho SAMPLE_PER_CONDITION casos por cada combinación
    (size, mode), tomando los primeros q_dir en orden (q001, q002, ...) para
    que la muestra sea estable y reproducible entre ejecuciones.
    """
    files = []
    for size_dir in sorted(DATA_DIR.glob("*")):
        if not size_dir.is_dir():
            continue
        for mode_dir in sorted(size_dir.glob("*")):
            if not mode_dir.is_dir():
                continue
            taken = 0
            for rep_dir in sorted(mode_dir.glob("rep*")):
                for q_dir in sorted(rep_dir.glob("q*")):
                    if taken >= SAMPLE_PER_CONDITION:
                        break
                    matches = list(q_dir.glob("decision_*.json"))
                    if matches:
                        files.append((size_dir.name, mode_dir.name, q_dir.name, matches[0]))
                        taken += 1
                if taken >= SAMPLE_PER_CONDITION:
                    break
    return files


def load_case(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    topic = str(data.get("topic", "")).strip()
    consensus = str(data.get("consensus_response", data.get("response", ""))).strip()
    reference_context = str(
        data.get("reference_context", "Context not provided. Evaluate on universal physics.")
    ).strip()
    return topic, consensus, reference_context


def main():
    cases = find_decision_files()
    print(f"📋 {len(cases)} debates encontrados para el estudio.\n")

    fieldnames = ["Size", "Mode", "Q_ID"] + list(ORDER_VARIANTS.keys()) + ["Range", "StdDev"]
    rows = []

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for i, (size, mode, q_id, path) in enumerate(cases, start=1):
            topic, consensus, reference_context = load_case(path)
            if not topic or not consensus:
                print(f"⚠️ [{i}/{len(cases)}] {size}/{mode}/{q_id}: sin topic/response, se salta")
                continue

            print(f"👉 [{i}/{len(cases)}] {size}/{mode}/{q_id}")
            row = {"Size": size, "Mode": mode, "Q_ID": q_id}
            scores = {}

            for order_name, order in ORDER_VARIANTS.items():
                result = evaluate_once(topic, consensus, reference_context, order=order)
                score = result.get("final_score") if result else None
                scores[order_name] = score
                row[order_name] = score
                print(f"     {order_name}: {score}")
                time.sleep(2)

            valid_scores = [s for s in scores.values() if s is not None]
            if len(valid_scores) >= 2:
                row["Range"] = round(max(valid_scores) - min(valid_scores), 2)
                row["StdDev"] = round(statistics.stdev(valid_scores), 2)
            else:
                row["Range"] = ""
                row["StdDev"] = ""

            writer.writerow(row)
            f_csv.flush()
            rows.append(row)

    print("\n🎉 Estudio completado.")
    print(f"📄 Resultados: {OUTPUT}\n")

    # --- Resumen: media/desviación por orden, y estabilidad media ---
    print("=== Media de nota final por orden ===")
    for order_name in ORDER_VARIANTS:
        vals = [r[order_name] for r in rows if r.get(order_name) is not None]
        if vals:
            print(f"  {order_name}: media={statistics.mean(vals):.2f}  n={len(vals)}")

    ranges = [r["Range"] for r in rows if r.get("Range") not in ("", None)]
    if ranges:
        print(f"\n=== Rango medio entre las 4 variantes (por debate) ===")
        print(f"  Media del rango: {statistics.mean(ranges):.2f}")
        print(f"  Máximo rango observado: {max(ranges):.2f}")


if __name__ == "__main__":
    main()
