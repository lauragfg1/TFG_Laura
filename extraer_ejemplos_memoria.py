"""
Extrae del corpus de resultados ya generado:
  1. Un ejemplo representativo de pregunta + respuesta de consenso, de la
     condicion de mejor calidad (70B heterogeneo), para ilustrar el
     capitulo de Implementacion/Evaluacion.
  2. Un ejemplo real de alucinacion "High" (con la justificacion que dio el
     juez), para la nota que pide el tutor en la Tabla 5.2.

No repite ninguna evaluacion: solo lee los decision_*.json y
evaluation_consistent_*.json que ya existen en 03_Langgraph_Parallel/data/.

Uso:
    python extraer_ejemplos_memoria.py
Salida:
    ejemplos_memoria.md  (listo para copiar a la memoria)
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "03_Langgraph_Parallel" / "data"
OUTPUT = BASE / "ejemplos_memoria.md"

REPS = (1, 2, 3)

# Orden de preferencia para buscar el ejemplo de alta alucinacion: se busca
# primero en modelos pequenos (donde es mas frecuente, ver Tabla 5.5) antes
# de caer a cualquier condicion.
ORDEN_BUSQUEDA_HALLUC = [
    ("2B", "homogeneo"), ("2B", "heterogeneo"), ("2B", "individual"),
    ("8B", "homogeneo"), ("8B", "heterogeneo"), ("8B", "individual"),
    ("70B", "individual"), ("70B", "homogeneo"), ("70B", "heterogeneo"),
]


def load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_decision_file(q_dir: Path):
    matches = list(q_dir.glob("decision_*.json")) + list(q_dir.glob("*_decision.json"))
    return matches[0] if matches else None


def iter_cases(size, arch):
    """Itera (rep, q_dir, decision_data, eval_data) para una condicion dada."""
    cond_dir = DATA_DIR / size / arch
    for rep in REPS:
        rep_dir = cond_dir / f"rep{rep}"
        if not rep_dir.exists():
            continue
        for q_dir in sorted(rep_dir.glob("q*")):
            eval_files = list(q_dir.glob("evaluation_consistent_*.json"))
            if not eval_files:
                continue
            eval_data = load_json(eval_files[0])
            decision_path = find_decision_file(q_dir)
            decision_data = load_json(decision_path) if decision_path else None
            if eval_data and decision_data:
                yield rep, q_dir, decision_data, eval_data


def buscar_ejemplo_calidad(size="70B", arch="heterogeneo", min_score=7.0):
    """Mejor ejemplo (Final_Score mas alto) de la condicion indicada."""
    mejor = None
    for rep, q_dir, decision, evaluacion in iter_cases(size, arch):
        if str(evaluacion.get("reliable", True)).strip().lower() not in ("true", "1"):
            continue
        score = float(evaluacion.get("final_score", 0) or 0)
        if score < min_score:
            continue
        if mejor is None or score > mejor["score"]:
            mejor = {
                "score": score, "rep": rep, "q_dir": q_dir.name,
                "decision": decision, "evaluacion": evaluacion,
                "size": size, "arch": arch,
            }
    return mejor


def buscar_ejemplo_alucinacion_alta():
    for size, arch in ORDEN_BUSQUEDA_HALLUC:
        for rep, q_dir, decision, evaluacion in iter_cases(size, arch):
            if str(evaluacion.get("hallucination_level", "")).strip().title() == "High":
                return {
                    "rep": rep, "q_dir": q_dir.name, "decision": decision,
                    "evaluacion": evaluacion, "size": size, "arch": arch,
                }
    return None


def fmt_ejemplo_calidad(ej):
    d, e = ej["decision"], ej["evaluacion"]
    ruta = f"03_Langgraph_Parallel/data/{ej['size']}/{ej['arch']}/rep{ej['rep']}/{ej['q_dir']}/"
    return f"""## Ejemplo de pregunta y respuesta de consenso (calidad alta)

**Condición:** LangGraph, modelo {ej['size']}, arquitectura {ej['arch']} — {ej['q_dir']} (rep{ej['rep']})
**Fuente:** `{ruta}`

**Pregunta (topic):**
> {d.get('topic', '').strip()}

**Respuesta de consenso (consensus_response):**
> {d.get('consensus_response', d.get('response', '')).strip()}

**Contexto de referencia recuperado por RAG (reference_context):**
> {d.get('reference_context', '').strip()}

**Evaluación del juez:**
- Precisión técnica: {e.get('technical_accuracy')}
- Adherencia al contexto: {e.get('context_adherence')}
- Capacidad de decisión: {e.get('decision_capability')}
- Nivel de alucinación: {e.get('hallucination_level')}
- Puntuación final: {e.get('final_score')}
- Justificación del juez: {e.get('justification', '').strip()}
"""


def fmt_ejemplo_alucinacion(ej):
    d, e = ej["decision"], ej["evaluacion"]
    ruta = f"03_Langgraph_Parallel/data/{ej['size']}/{ej['arch']}/rep{ej['rep']}/{ej['q_dir']}/"
    return f"""## Ejemplo de alucinación "Alta" (para la nota de la Tabla 5.2)

**Condición:** LangGraph, modelo {ej['size']}, arquitectura {ej['arch']} — {ej['q_dir']} (rep{ej['rep']})
**Fuente:** `{ruta}`

**Pregunta (topic):**
> {d.get('topic', '').strip()}

**Respuesta de consenso (consensus_response):**
> {d.get('consensus_response', d.get('response', '')).strip()}

**Contexto de referencia recuperado por RAG (reference_context):**
> {d.get('reference_context', '').strip()}

**Por qué el juez la marcó como alucinación alta:**
- Precisión técnica: {e.get('technical_accuracy')}
- Adherencia al contexto: {e.get('context_adherence')}
- Capacidad de decisión: {e.get('decision_capability')}
- Puntuación base: {e.get('base_score')} → Puntuación final tras penalización: {e.get('final_score')}
- Justificación del juez: {e.get('justification', '').strip()}
"""


def main():
    partes = ["# Ejemplos extraídos automáticamente para la memoria\n"]

    ejemplo_calidad = buscar_ejemplo_calidad()
    if not ejemplo_calidad:
        # Si no hay ninguno >= 7.0 en 70B heterogeneo, relajamos el umbral.
        ejemplo_calidad = buscar_ejemplo_calidad(min_score=0.0)
    if ejemplo_calidad:
        partes.append(fmt_ejemplo_calidad(ejemplo_calidad))
        print(f"✓ Ejemplo de calidad: {ejemplo_calidad['size']}/{ejemplo_calidad['arch']}/"
              f"rep{ejemplo_calidad['rep']}/{ejemplo_calidad['q_dir']} "
              f"(score={ejemplo_calidad['score']})")
    else:
        print("⚠️  No se encontró ningún ejemplo de calidad.")

    ejemplo_halluc = buscar_ejemplo_alucinacion_alta()
    if ejemplo_halluc:
        partes.append(fmt_ejemplo_alucinacion(ejemplo_halluc))
        print(f"✓ Ejemplo de alucinación alta: {ejemplo_halluc['size']}/{ejemplo_halluc['arch']}/"
              f"rep{ejemplo_halluc['rep']}/{ejemplo_halluc['q_dir']}")
    else:
        print("⚠️  No se encontró ningún ejemplo con Hallucination_Level=High.")

    OUTPUT.write_text("\n\n".join(partes), encoding="utf-8")
    print(f"\nGuardado en: {OUTPUT}")


if __name__ == "__main__":
    main()
