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
    os.path.join(os.path.dirname(__file__), '..', '01_Langgraph_Debate')
))
from qdrant_rag import recuperar_contexto
from dataset_questions import load_all_questions
from llm_config import NUM_PREDICT, TEMPERATURE, STOP_SEQUENCES

from crewai import Agent, Task
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
# get_openai_callback captura prompt_tokens y completion_tokens de todas las
# llamadas LLM realizadas dentro de su bloque, sin modificar el comportamiento del Crew
from langchain_community.callbacks import get_openai_callback

# Ollama expone un endpoint compatible con la API de OpenAI
BASE_URL = "http://localhost:11434/v1"

# Número de rondas de debate DLB↔PNM, igual que en LangGraph (max_rounds=5) y AutoGen (N_ROUNDS=5)
N_ROUNDS = 5

# Misma asignacion de modelos por tamaño que LangGraph (nodes.py::get_models,
# modo heterogeneo) para que la comparacion entre frameworks sea comparable.
MODELS = {
    "8B": {"mod": "llama3.1:8b", "dlb": "qwen2.5:7b", "pnm": "mistral:7b"},
    "2B": {"mod": "gemma2:2b", "dlb": "qwen2.5:1.5b", "pnm": "deepseek-r1:1.5b"},
}
SIZE = "8B"  # sobrescrito por --size en main()


def make_llm(model: str) -> ChatOpenAI:
    """Crea un cliente LLM apuntando al servidor Ollama local.

    Usa la misma temperatura/max_tokens/stop que LangGraph y AutoGen
    (llm_config.py), para que el mismo modelo no se compare bajo un muestreo
    distinto según el framework.
    """
    # "stop" no se puede fijar aquí: CrewAI ya pasa su propio `stop` en cada
    # llamada interna (parseo ReAct), y langchain_openai revienta con
    # "`stop` found in both the input and default params" si además viene
    # precargado en el cliente. A diferencia de LangGraph/AutoGen (transcript
    # compartido entre agentes, donde StopSequences evita que uno impersone al
    # otro), en CrewAI cada agente responde a su propia Task aislada, así que
    # el riesgo de impersonación que motivó STOP_SEQUENCES es menor aquí.
    return ChatOpenAI(
        model=model,
        base_url=BASE_URL,
        api_key="NotRequired",
        temperature=TEMPERATURE,
        max_tokens=NUM_PREDICT,
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
            f"Reference Context: {context}\n\n"
            "Drive discussion, prevent repetition, push for deeper understanding. "
            "Maintain technical precision and neutral tone. "
            "Max 100 words per intervention."
        ),
        verbose=False,
        allow_delegation=False,
        llm=make_llm(MODELS[SIZE]["mod"])
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
        llm=make_llm(MODELS[SIZE]["dlb"])
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
        llm=make_llm(MODELS[SIZE]["pnm"])
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
        llm=make_llm(MODELS[SIZE]["mod"])
    )

    return moderador, dlb_expert, pnm_expert, juez


_REACT_FAILURE = "Agent stopped due to iteration limit or time limit."


def execute_task(task: Task, agent: Agent, context: str = None) -> str:
    """Ejecuta una Task con fallback si el parseo ReAct de CrewAI falla.

    CrewAI 0.1.24 envuelve cada Agent en un CrewAgentExecutor de LangChain
    (bind=stop=["\\nObservation"], parseo tipo ReAct que busca un marcador
    "Final Answer:" en la salida) aunque no usemos herramientas. Los modelos
    de 8B lo siguen bien; los de 2B (gemma2:2b, deepseek-r1:1.5b, qwen2.5:1.5b)
    no siempre emiten ese marcador exacto, y el executor reintenta hasta
    max_iterations=15 veces antes de rendirse con _REACT_FAILURE -- un solo
    debate de prueba en 2B tardo 552s y disparo 330k tokens de entrada por
    esto. Como no usamos tools, el bucle ReAct no aporta nada; si falla,
    reintentamos con una llamada directa al LLM del agente (mismo
    backstory+description+context, sin el envoltorio ReAct).

    Para SIZE != "8B" se salta directamente a la llamada directa: dejar que
    el executor agote sus hasta 15 reintentos fallidos antes de caer al
    fallback multiplicaba el tiempo por ~4x sin cambiar el resultado (el
    fallback siempre acababa disparandose igual). Para 8B se mantiene el
    camino original (probado con los 150 debates ya evaluados), sin tocarlo.
    """
    def _direct_call():
        prompt = f"{agent.backstory}\n\n{task.description}"
        if context:
            prompt += f"\n\nContext:\n{context}"
        response = agent.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    if SIZE != "8B":
        return _direct_call()

    result = task.execute(context)
    if _REACT_FAILURE not in result:
        return result
    return _direct_call()


def lanzar_debate(topic: str, context: str, n_rounds: int = N_ROUNDS):
    """
    Ejecuta el debate como una secuencia de llamadas directas Task.execute(context=...).

    Estructura (16 llamadas LLM para n_rounds=5):
      tarea_intro(mod) → dlb_1 → pnm_1 → mod_1 → dlb_2 → pnm_2 → mod_2
                       → dlb_3 → pnm_3 → mod_3 → dlb_4 → pnm_4 → mod_4
                       → dlb_5 → pnm_5 → tarea_sintesis(juez)

    NOTA IMPORTANTE (bug corregido): la version anterior construia cada Task
    con `context=[tarea_previa]`, asumiendo que CrewAI encadenaria el output
    real de esas tareas. Pero `crewai==0.1.24` (la version instalada aqui) no
    tiene ningun campo `context` en su clase Task -- pydantic lo descarta en
    silencio, sin error. Y `Crew.kickoff()` (Process.sequential) tampoco lee
    ese campo: solo reenvia el output de la tarea INMEDIATAMENTE anterior a la
    siguiente (ver crewai/crew.py, __sequential_loop). Resultado verificado en
    un debate real: las 5 rondas de un mismo experto salian casi identicas,
    porque nunca veian de verdad lo que respondia el otro agente ni el
    moderador -- cada Task solo tenia su propia description estatica.
    Aqui se llama a `Task.execute(context=...)` directamente (el metodo real
    y funcional en esta version, ver crewai/task.py), pasando a mano el texto
    de las respuestas relevantes de la ronda -- igual que el historial
    ventana-deslizante de LangGraph (nodes.py, RECENT_WINDOW).

    De paso, se ha quitado el "Context: {context}" (RAG completo) que se
    repetia en la description de cada tarea -- ya esta en el backstory de
    cada agente (crear_agentes), asi que era pura duplicacion de tokens.
    """
    moderador, dlb_expert, pnm_expert, juez = crear_agentes(topic, context)

    # get_openai_callback intercepta todas las llamadas al endpoint /v1 durante el
    # debate y acumula prompt_tokens/completion_tokens reportados por Ollama
    # (equivalentes a prompt_eval_count/eval_count de la API nativa que usa LangGraph).
    # Sigue funcionando igual llamando a Task.execute() directamente, no depende de Crew.
    t_start = time.perf_counter()
    with get_openai_callback() as cb:
        # Tarea 1: el moderador introduce el problema técnico y abre el debate
        tarea_intro = Task(
            description="Introduce the technical problem clearly. Invite both experts to discuss. Max 100 words.",
            expected_output="Brief technical introduction of the debate topic.",
            agent=moderador
        )
        intro_out = execute_task(tarea_intro, moderador)

        dlb_rounds, pnm_rounds, mod_rounds = [], [], []
        last_moderator_msg = intro_out
        last_pnm_msg = None

        for i in range(n_rounds):
            ronda = f"[Round {i+1}/{n_rounds}]"

            # DLB ve la ultima guia del moderador y la ultima respuesta de PNM (si la hay)
            t_dlb = Task(
                description=(
                    f"{ronda} Provide your DLB physical/budget analysis reacting to the "
                    "context below. New insight only, no repetition."
                ),
                expected_output=f"DLB technical analysis round {i+1}. Max 100 words.",
                agent=dlb_expert,
            )
            dlb_ctx = f"Moderator's latest message:\n{last_moderator_msg}"
            if last_pnm_msg:
                dlb_ctx += f"\n\nPNM's previous analysis:\n{last_pnm_msg}"
            dlb_out = execute_task(t_dlb, dlb_expert, dlb_ctx)
            dlb_rounds.append(dlb_out)

            # PNM lee el analisis de DLB de esta misma ronda y lo complementa o refuta
            t_pnm = Task(
                description=(
                    f"{ronda} Read DLB's analysis below and provide your PNM network/payload "
                    "analysis. Counter or complement. New insight only."
                ),
                expected_output=f"PNM technical analysis round {i+1}. Max 100 words.",
                agent=pnm_expert,
            )
            pnm_out = execute_task(t_pnm, pnm_expert, f"DLB's analysis this round:\n{dlb_out}")
            pnm_rounds.append(pnm_out)
            last_pnm_msg = pnm_out

            # El moderador guía la transición entre rondas (no se añade tras la última ronda)
            if i < n_rounds - 1:
                t_mod = Task(
                    description=(
                        f"{ronda} Read DLB and PNM analyses below and guide the next round. "
                        "Summarize key points, prevent repetition, push for deeper understanding. "
                        "Max 100 words."
                    ),
                    expected_output=f"Moderator guidance for round {i+2}. Max 100 words.",
                    agent=moderador,
                )
                mod_out = execute_task(t_mod, moderador, f"DLB:\n{dlb_out}\n\nPNM:\n{pnm_out}")
                mod_rounds.append(mod_out)
                last_moderator_msg = mod_out

        # Tarea final: el juez lee el debate completo (pasado como contexto real) y sintetiza
        partes = [f"[Moderator intro]\n{intro_out}"]
        for i in range(n_rounds):
            partes.append(f"[Round {i+1}] DLB: {dlb_rounds[i]}")
            partes.append(f"[Round {i+1}] PNM: {pnm_rounds[i]}")
            if i < len(mod_rounds):
                partes.append(f"[Round {i+1}→{i+2}] Moderator: {mod_rounds[i]}")
        debate_transcript = "\n\n".join(partes)

        tarea_sintesis = Task(
            description="Read the complete debate below and produce the JSON decision.",
            expected_output="Valid JSON with topic, winner, consensus_response, reference_context.",
            agent=juez,
        )
        judge_out = execute_task(tarea_sintesis, juez, debate_transcript)

    t_total = time.perf_counter() - t_start
    tokens_in = cb.prompt_tokens
    tokens_out = cb.completion_tokens

    historial = {
        "Moderator_intro": intro_out,
        "DLB_rounds": "\n\n".join(f"[Round {i+1}] {t}" for i, t in enumerate(dlb_rounds)),
        "PNM_rounds": "\n\n".join(f"[Round {i+1}] {t}" for i, t in enumerate(pnm_rounds)),
        "Moderator_rounds": "\n\n".join(f"[Round {i+1}→{i+2}] {t}" for i, t in enumerate(mod_rounds)),
        "Judge": judge_out,
    }

    return judge_out, historial, t_total, tokens_in, tokens_out


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
            base_name, SIZE, "heterogeneo", winner,
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

    Resultados guardados en: 03_CrewAI_Debate/Resultados/repN/ (--rep, default 1)
      - q001/ ... q050/  → TXT + JSON por debate
      - metricas_crewai.csv → métricas agregadas de esa repetición
    """
    global SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1,
                        help="Número de pregunta desde la que empezar (1-indexed, default=1)")
    parser.add_argument("--rep", type=int, default=1,
                        help="Número de repetición del experimento (crea Resultados/repN/)")
    parser.add_argument("--size", choices=list(MODELS.keys()), default="8B",
                        help="Tamaño de los modelos (misma asignación que LangGraph)")
    args = parser.parse_args()
    start_idx = max(0, args.start - 1)  # convertir a 0-indexed
    SIZE = args.size

    # Una carpeta por repetición, para no sobrescribir las anteriores.
    # 8B mantiene la ruta original (Resultados/repN/) por compatibilidad con
    # los datos ya evaluados; otros tamaños van en su propia subcarpeta.
    if args.size == "8B":
        resultados_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'Resultados', f'rep{args.rep}')
        )
    else:
        resultados_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'Resultados', args.size, f'rep{args.rep}')
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
