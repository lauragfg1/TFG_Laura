"""
TFG - Comparativa de Frameworks Multi-Agente para Debates Técnicos en SatCom
Implementación con CrewAI (Process.sequential)

Estructura del debate (equivalente a LangGraph y AutoGen):
  - 4 agentes: Moderador, DLB Expert, PNM Expert, Juez
  - 5 rondas de debate con moderador activo entre rondas
  - 16 llamadas LLM totales: 1 intro + 4 mod_inter + 5 DLB + 5 PNM + 1 Juez
  - Métricas de tokens exactas vía get_openai_callback() (endpoint OpenAI-compatible /v1)

Modelos (modo 8B heterogéneo, igual que LangGraph y AutoGen):
  - Moderador / Juez: llama3.1:8b
  - DLB Expert:       qwen2.5:7b
  - PNM Expert:       mistral:7b
"""

import os
import sys
import time
import json
import csv
import re
import argparse

# Acceso al módulo RAG compartido con LangGraph
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '03_Langgraph_Parallel')
))
from qdrant_rag import recuperar_contexto
from dataset_questions import load_all_questions
from llm_config import NUM_PREDICT, TEMPERATURE, STOP_SEQUENCES

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
# get_openai_callback captura prompt_tokens y completion_tokens de todas las
# llamadas LLM realizadas dentro de su bloque, sin modificar el comportamiento del Crew
from langchain_community.callbacks import get_openai_callback

# Ollama expone un endpoint compatible con la API de OpenAI
BASE_URL = "http://localhost:11434/v1"

# Número de rondas de debate DLB↔PNM, igual que en LangGraph (max_rounds=5) y AutoGen (N_ROUNDS=5)
N_ROUNDS = 5


def make_llm(model: str) -> ChatOpenAI:
    """Crea un cliente LLM apuntando al servidor Ollama local.

    Usa la misma temperatura/max_tokens/stop que LangGraph y AutoGen
    (llm_config.py), para que el mismo modelo no se compare bajo un muestreo
    distinto según el framework.
    """
    return ChatOpenAI(
        model=model,
        base_url=BASE_URL,
        api_key="NotRequired",
        temperature=TEMPERATURE,
        max_tokens=NUM_PREDICT,
        stop=STOP_SEQUENCES,
    )


def crear_agentes(topic: str, context: str):
    """
    Define los 4 agentes del debate. El contexto RAG se inyecta en el backstory
    de cada agente para que esté disponible en todas sus tareas.
    """
    moderador = Agent(
        role="Moderator",
        goal="Drive the debate forward with precision and neutrality.",
        backstory=(
            f"You moderate a technical debate on: {topic}.\n"
            "Drive discussion, prevent repetition, push for deeper understanding. "
            "Maintain technical precision and neutral tone. "
            "Max 100 words per intervention."
        ),
        verbose=False,
        allow_delegation=False,
        llm=make_llm("llama3.1:8b")
    )

    # Experto en presupuesto de enlace (DLB = Dynamic Link Budget)
    dlb_expert = Agent(
        role="DLB_Expert",
        goal="Analyze physical mechanisms and link budget implications.",
        backstory=(
            f"Topic: {topic}\nReference Context: {context}\n\n"
            "Focus ONLY on physical mechanism and link budget implications. "
            "Explain array factor, interference, spacing effects. "
            "Do NOT repeat generic textbook statements — provide detailed, specific analysis based on the Context. "
            "Focus on what is the cost in terms of performance, capacity, interference of design choices. "
            "DO NOT invent metrics not in the Reference Context. If the Context lacks data, explicitly state so and rely strictly on universal physics principles. "
            "Provide a NEW insight each round. Max 100 words."
        ),
        verbose=False,
        allow_delegation=False,
        llm=make_llm("qwen2.5:7b")
    )

    # Experto en gestión de red y payload (PNM = Payload & Network Management)
    pnm_expert = Agent(
        role="PNM_Expert",
        goal="Analyze payload and network management implications.",
        backstory=(
            f"Topic: {topic}\nReference Context: {context}\n\n"
            "Focus ONLY on payload and network implications. "
            "Describe cause-effect relationships (e.g., more interference → lower capacity). "
            "Provide specific technical deductions. Avoid vague generalizations. "
            "DO NOT use numbers unless explicitly in the Reference Context. If the Context lacks specifics, base your reasoning purely on universal networking/telecommunications principles without making up data. "
            "Provide a NEW insight each round. Max 100 words."
        ),
        verbose=False,
        allow_delegation=False,
        llm=make_llm("mistral:7b")
    )

    # El juez sintetiza el debate completo y determina el ganador
    juez = Agent(
        role="Judge",
        goal="Produce the final technical synthesis as JSON.",
        backstory=(
            f"You synthesize a technical debate on: {topic}\n"
            f"Reference Context: {context}\n\n"
            "Merge DLB (physical layer) and PNM (system impact) insights. "
            "The first sentence of consensus_response MUST directly answer the topic question. "
            "DO NOT fabricate data, numbers, or metrics not in the debate or context. "
            "Return ONLY valid JSON:\n"
            "{\n"
            f'   "topic": "{topic}",\n'
            '    "winner": "DLB or PNM",\n'
            '    "consensus_response": "Write your deep, technical, cause-effect explanation here merging DLB and PNM insights. Max 150 words.",\n'
            '    "reference_context": "60-word factual summary of the reference context. Do NOT copy expert chat history."\n'
            "}"
        ),
        verbose=False,
        allow_delegation=False,
        llm=make_llm("llama3.1:8b")
    )

    return moderador, dlb_expert, pnm_expert, juez


def lanzar_debate(topic: str, context: str, n_rounds: int = N_ROUNDS):
    """
    Construye y ejecuta el debate como una cadena de tareas secuenciales (Process.sequential).

    Estructura de tareas (16 llamadas LLM para n_rounds=5):
      tarea_intro(mod) → dlb_1 → pnm_1 → mod_1 → dlb_2 → pnm_2 → mod_2
                       → dlb_3 → pnm_3 → mod_3 → dlb_4 → pnm_4 → mod_4
                       → dlb_5 → pnm_5 → tarea_sintesis(juez)

    Cada tarea recibe el output de la anterior como contexto, formando una
    cadena de argumentación progresiva equivalente al grafo de LangGraph.
    """
    moderador, dlb_expert, pnm_expert, juez = crear_agentes(topic, context)

    # Tarea 1: el moderador introduce el problema técnico y abre el debate
    tarea_intro = Task(
        description=(
            f"Introduce the technical problem clearly.\n"
            f"Topic: {topic}\nContext: {context}\n"
            "Invite both experts to discuss. Max 100 words."
        ),
        expected_output="Brief technical introduction of the debate topic.",
        agent=moderador
    )

    tareas = [tarea_intro]
    tareas_dlb = []       # referencias para reconstruir el historial al final
    tareas_pnm = []
    tareas_mod_inter = []  # moderadores entre rondas (n_rounds - 1 tareas)

    for i in range(n_rounds):
        ronda = f"[Round {i+1}/{n_rounds}]"

        # DLB recibe como contexto la tarea anterior (intro en ronda 1, moderador en rondas 2-5)
        t_dlb = Task(
            description=(
                f"{ronda} Topic: {topic}. Context: {context}.\n"
                "Provide your DLB physical/budget analysis. "
                "New insight only, no repetition."
            ),
            expected_output=f"DLB technical analysis round {i+1}. Max 100 words.",
            agent=dlb_expert,
            context=[tareas[-1]]
        )

        # PNM recibe el análisis de DLB de esta misma ronda y lo complementa o refuta
        t_pnm = Task(
            description=(
                f"{ronda} Topic: {topic}. Context: {context}.\n"
                "Read DLB's analysis and provide your PNM network/payload analysis. "
                "Counter or complement. New insight only."
            ),
            expected_output=f"PNM technical analysis round {i+1}. Max 100 words.",
            agent=pnm_expert,
            context=[t_dlb]
        )

        tareas.extend([t_dlb, t_pnm])
        tareas_dlb.append(t_dlb)
        tareas_pnm.append(t_pnm)

        # El moderador guía la transición entre rondas (no se añade tras la última ronda)
        # Equivale a las llamadas intermedias del moderador en LangGraph y AutoGen
        if i < n_rounds - 1:
            t_mod = Task(
                description=(
                    f"{ronda} Read DLB and PNM analyses and guide the next round.\n"
                    f"Topic: {topic}.\n"
                    "Summarize key points, prevent repetition, push for deeper understanding. "
                    "Max 100 words."
                ),
                expected_output=f"Moderator guidance for round {i+2}. Max 100 words.",
                agent=moderador,
                context=[t_dlb, t_pnm]
            )
            tareas.append(t_mod)
            tareas_mod_inter.append(t_mod)

    # Tarea final: el juez lee todo el historial y produce la síntesis en JSON
    tarea_sintesis = Task(
        description=(
            f"Read the complete {n_rounds}-round debate and produce the JSON decision.\n"
            f"Topic: {topic}. Context: {context}."
        ),
        expected_output="Valid JSON with topic, winner, consensus_response, reference_context.",
        agent=juez,
        context=tareas  # recibe todas las tareas anteriores como contexto
    )
    tareas.append(tarea_sintesis)

    crew = Crew(
        agents=[moderador, dlb_expert, pnm_expert, juez],
        tasks=tareas,
        verbose=False,
        process=Process.sequential
    )

    # get_openai_callback intercepta todas las llamadas al endpoint /v1 durante el kickoff
    # y acumula prompt_tokens y completion_tokens reportados por Ollama (valores exactos,
    # equivalentes a prompt_eval_count y eval_count de la API nativa usada por LangGraph)
    t_start = time.perf_counter()
    with get_openai_callback() as cb:
        resultado = crew.kickoff()
    t_total = time.perf_counter() - t_start

    tokens_in = cb.prompt_tokens      # tokens de entrada acumulados en todas las llamadas
    tokens_out = cb.completion_tokens  # tokens generados acumulados en todas las llamadas

    # Reconstruir historial por rol para guardarlo en TXT
    historial = {
        "Moderator_intro": get_task_output(tarea_intro),
        "DLB_rounds": "\n\n".join(
            f"[Round {i+1}] {get_task_output(t)}"
            for i, t in enumerate(tareas_dlb)
        ),
        "PNM_rounds": "\n\n".join(
            f"[Round {i+1}] {get_task_output(t)}"
            for i, t in enumerate(tareas_pnm)
        ),
        "Moderator_rounds": "\n\n".join(
            f"[Round {i+1}→{i+2}] {get_task_output(t)}"
            for i, t in enumerate(tareas_mod_inter)
        ),
        "Judge": get_task_output(tarea_sintesis)
    }

    return resultado, historial, t_total, tokens_in, tokens_out


def get_task_output(task) -> str:
    """
    Obtiene el texto de salida de una tarea CrewAI de forma robusta.
    Según la versión de CrewAI, el output puede estar en .raw, .result,
    o dentro del str() del objeto como campo result='...'.
    """
    output = task.output
    if output is None:
        return ""
    raw = getattr(output, 'raw', None)
    if raw:
        return str(raw)
    result_attr = getattr(output, 'result', None)
    if result_attr:
        return str(result_attr)
    # Fallback: extraer el campo result= del str() y decodificar escapes
    full = str(output)
    match = re.search(r"result='(.*)'$", full, re.DOTALL)
    if match:
        return match.group(1).encode('raw_unicode_escape').decode('unicode_escape')
    return full


def extract_json(text: str):
    """Extrae el primer bloque JSON válido del texto de respuesta del juez."""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None


def guardar_resultado(q_id, topic, context, historial,
                      resultado_final, t_total, tokens_in, tokens_out, resultados_dir):
    """
    Persiste los resultados de un debate en tres formatos:
      - TXT: historial completo del debate por rol
      - JSON: decisión estructurada del juez (winner, consensus_response, reference_context)
      - CSV: métricas de rendimiento (mismo esquema que LangGraph y AutoGen)

    Esquema CSV (igual en los tres frameworks):
      ID, Size, Mode, Winner, Total_S, Load_S, Prompt_S, Gen_S,
      Tokens_In, Tokens_Out, Avg_TPS, Topic

    Nota: Load_S, Prompt_S y Gen_S son 0.0 porque el endpoint OpenAI-compatible
    de Ollama (/v1) no expone el desglose interno de tiempos. LangGraph los obtiene
    porque usa la API nativa de Ollama (/api/chat) directamente.
    """
    q_dir = os.path.join(resultados_dir, q_id)
    os.makedirs(q_dir, exist_ok=True)
    base_name = f"{q_id}_debateC"

    # --- TXT con el historial completo del debate ---
    txt_path = os.path.join(q_dir, f"{base_name}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"TOPIC: {topic}\n{'='*60}\n")
        f.write(f"CONTEXT:\n{context}\n{'='*60}\n")
        for role, text in historial.items():
            f.write(f"[{role}]:\n{text}\n{'-'*40}\n")

    # --- JSON con la decisión estructurada del juez ---
    judge_text = historial.get("Judge", str(resultado_final))
    judge_data = extract_json(judge_text)
    winner = "Unknown"
    if judge_data:
        json_path = os.path.join(q_dir, f"{base_name}_decision.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(judge_data, f, indent=4)
        winner = judge_data.get("winner", "Unknown")

    # Avg_TPS: tokens generados por segundo (throughput global del debate)
    avg_tps = tokens_out / t_total if t_total > 0 else 0

    # --- CSV con métricas de rendimiento ---
    csv_path = os.path.join(resultados_dir, "metricas_crewai.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "ID", "Size", "Mode", "Winner", "Total_S",
                "Load_S", "Prompt_S", "Gen_S",
                "Tokens_In", "Tokens_Out", "Avg_TPS", "Topic"
            ])
        writer.writerow([
            base_name, "8B", "heterogeneo", winner,
            round(t_total, 3),
            0.0,   # Load_S: no disponible en endpoint /v1
            0.0,   # Prompt_S: no disponible en endpoint /v1
            0.0,   # Gen_S: no disponible en endpoint /v1
            tokens_in, tokens_out,
            round(avg_tps, 2), topic
        ])


def main():
    """
    Ejecuta los 50 debates del dataset en orden secuencial.

    Las preguntas se cargan con dataset_questions.load_all_questions(), la misma
    fuente que usan LangGraph y AutoGen, así que los tres frameworks procesan
    exactamente las mismas 50 preguntas en el mismo orden.

    Resultados guardados en: 05_CrewAI_Debate/Resultados/repN/ (--rep, default 1)
      - q001/ ... q050/  → TXT + JSON por debate
      - metricas_crewai.csv → métricas agregadas de esa repetición
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1,
                        help="Número de pregunta desde la que empezar (1-indexed, default=1)")
    parser.add_argument("--rep", type=int, default=1,
                        help="Número de repetición del experimento (crea Resultados/repN/)")
    args = parser.parse_args()
    start_idx = max(0, args.start - 1)  # convertir a 0-indexed

    # Una carpeta por repetición, para no sobrescribir las anteriores
    resultados_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'Resultados', f'rep{args.rep}')
    )
    dataset_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'Dataset_Preguntas')
    )
    os.makedirs(resultados_dir, exist_ok=True)

    # Misma fuente que LangGraph (dataset_questions.py): recorre los .txt del
    # dataset directamente en vez de depender de index.txt, para garantizar por
    # código las mismas 50 preguntas en el mismo orden en los tres frameworks.
    preguntas = load_all_questions(dataset_dir)

    print(f"Cargadas {len(preguntas)} preguntas. Repetición {args.rep}. Ejecutando desde q{args.start:03d} hasta q050...")

    for idx, topic in enumerate(preguntas[start_idx:50], start=start_idx):
        q_id = f"q{idx+1:03d}"
        print(f"\n{'*'*50}\n[{q_id}] {topic}\n{'*'*50}")

        # Recuperar contexto técnico relevante desde la base vectorial Qdrant
        try:
            context = recuperar_contexto(topic, topic=topic) or "N/A"
        except Exception as e:
            print(f"Error RAG: {e}")
            context = "Error retrieving context."

        # Aislada por pregunta (igual que LangGraph): si un debate falla, se
        # registra y se continúa con el resto en vez de abortar todo el run.
        try:
            resultado, historial, t_total, tokens_in, tokens_out = lanzar_debate(
                topic, context, n_rounds=N_ROUNDS
            )
            guardar_resultado(
                q_id, topic, context, historial,
                resultado, t_total, tokens_in, tokens_out, resultados_dir
            )
        except Exception as e:
            print(f"Error en debate {q_id}: {e}")

        # Pausa entre debates para no saturar la GPU
        time.sleep(1)


if __name__ == "__main__":
    main()
