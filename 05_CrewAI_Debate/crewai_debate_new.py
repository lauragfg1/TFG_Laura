import os
import sys
import time
import json
import csv
import re

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '03_Langgraph_Parallel')
))
from qdrant_rag import recuperar_contexto

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

BASE_URL = "http://localhost:11434/v1"
N_ROUNDS = 5  # Ciclos DLB+PNM equivalentes a LangGraph

def make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        base_url=BASE_URL,
        api_key="NotRequired",
        temperature=0.7
    )

def crear_agentes(topic: str, context: str):
    moderador = Agent(
        role="Moderator",
        goal="Drive the debate forward with precision and neutrality.",
        backstory=(
            f"You moderate a technical debate on: {topic}.\n"
            "Drive discussion, prevent repetition, push for deeper understanding. "
            "Max 100 words per intervention."
        ),
        verbose=False,
        allow_delegation=False,
        llm=make_llm("llama3.1:8b")
    )

    dlb_expert = Agent(
        role="DLB_Expert",
        goal="Analyze physical mechanisms and link budget implications.",
        backstory=(
            f"Topic: {topic}\nReference Context: {context}\n\n"
            "Focus ONLY on physical mechanism and link budget implications. "
            "Explain array factor, interference, spacing effects. "
            "DO NOT invent metrics not in the Reference Context. "
            "Provide a NEW insight each round. Max 100 words."
        ),
        verbose=False,
        allow_delegation=False,
        llm=make_llm("qwen2.5:7b")
    )

    pnm_expert = Agent(
        role="PNM_Expert",
        goal="Analyze payload and network management implications.",
        backstory=(
            f"Topic: {topic}\nReference Context: {context}\n\n"
            "Focus ONLY on payload and network implications. "
            "Describe cause-effect relationships. "
            "DO NOT use numbers unless explicitly in the Reference Context. "
            "Provide a NEW insight each round. Max 100 words."
        ),
        verbose=False,
        allow_delegation=False,
        llm=make_llm("mistral:7b")
    )

    juez = Agent(
        role="Judge",
        goal="Produce the final technical synthesis as JSON.",
        backstory=(
            f"You synthesize a technical debate on: {topic}\n"
            f"Reference Context: {context}\n\n"
            "Merge DLB and PNM insights. "
            "DO NOT fabricate data not in the debate or context. "
            "Return ONLY valid JSON:\n"
            "{\n"
            f'    "topic": "{topic}",\n'
            '    "winner": "DLB or PNM",\n'
            '    "consensus_response": "Max 150 words.",\n'
            '    "reference_context": "60-word factual summary."\n'
            "}"
        ),
        verbose=False,
        allow_delegation=False,
        llm=make_llm("llama3.1:8b")
    )

    return moderador, dlb_expert, pnm_expert, juez

def lanzar_debate(topic: str, context: str, n_rounds: int = N_ROUNDS):
    moderador, dlb_expert, pnm_expert, juez = crear_agentes(topic, context)

    # Tarea introductoria del moderador
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
    tareas_dlb = []
    tareas_pnm = []

    for i in range(n_rounds):
        ronda = f"[Round {i+1}/{n_rounds}]"

        # DLB recibe contexto de la tarea anterior
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

        # PNM recibe el argumento de DLB
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

    # Síntesis final — el juez recibe TODO el historial
    tarea_sintesis = Task(
        description=(
            f"Read the complete {n_rounds}-round debate and produce the JSON decision.\n"
            f"Topic: {topic}. Context: {context}."
        ),
        expected_output="Valid JSON with topic, winner, consensus_response, reference_context.",
        agent=juez,
        context=tareas
    )
    tareas.append(tarea_sintesis)

    crew = Crew(
        agents=[moderador, dlb_expert, pnm_expert, juez],
        tasks=tareas,
        verbose=False,
        process=Process.sequential
    )

    t_start = time.perf_counter()
    resultado = crew.kickoff()
    t_total = time.perf_counter() - t_start

    # Reconstruir historial
    historial = {
        "Moderator_intro": getattr(tarea_intro.output, 'raw', str(tarea_intro.output)),
        "DLB_rounds": "\n\n".join(
            f"[Round {i+1}] {getattr(t.output, 'raw', str(t.output))}"
            for i, t in enumerate(tareas_dlb)
        ),
        "PNM_rounds": "\n\n".join(
            f"[Round {i+1}] {getattr(t.output, 'raw', str(t.output))}"
            for i, t in enumerate(tareas_pnm)
        ),
        "Judge": getattr(tarea_sintesis.output, 'raw', str(tarea_sintesis.output))
    }

    return resultado, historial, t_total

def extract_json(text: str):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None

def contar_tokens(text: str) -> int:
    return max(1, int(len(str(text).split()) / 0.75))

def guardar_resultado(q_id, topic, context, historial, 
                      resultado_final, t_total, resultados_dir):
    q_dir = os.path.join(resultados_dir, q_id)
    os.makedirs(q_dir, exist_ok=True)
    base_name = f"{q_id}_debateC"

    # TXT
    txt_path = os.path.join(q_dir, f"{base_name}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"TOPIC: {topic}\n{'='*60}\n")
        f.write(f"CONTEXT:\n{context}\n{'='*60}\n")
        for role, text in historial.items():
            f.write(f"[{role}]:\n{text}\n{'-'*40}\n")

    # JSON
    judge_text = historial.get("Judge", str(resultado_final))
    judge_data = extract_json(judge_text)
    winner = "Unknown"
    if judge_data:
        json_path = os.path.join(q_dir, f"{base_name}_decision.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(judge_data, f, indent=4)
        winner = judge_data.get("winner", "Unknown")

    # Métricas aproximadas
    tokens_out = sum(contar_tokens(t) for t in historial.values())
    base_ctx = contar_tokens(context) + contar_tokens(topic)
    # CrewAI inyecta contexto en cada tarea: n_rounds * 2 tareas + intro + síntesis
    n_tareas = N_ROUNDS * 2 + 2
    tokens_in = base_ctx * n_tareas
    avg_tps = tokens_out / t_total if t_total > 0 else 0

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
            round(t_total, 3), 0.0, 0.0, round(t_total, 3),
            tokens_in, tokens_out,
            round(avg_tps, 2), topic
        ])

def main():
    resultados_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'Resultados')
    )
    dataset_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'Dataset_Preguntas')
    )
    os.makedirs(resultados_dir, exist_ok=True)

    index_path = os.path.join(dataset_dir, 'index.txt')
    with open(index_path, 'r', encoding='utf-8') as f:
        preguntas = [l.strip() for l in f if l.strip()]

    print(f"Cargadas {len(preguntas)} preguntas")

    for idx, topic in enumerate(preguntas[:50]):
        q_id = f"q{idx+1:03d}"
        print(f"\n{'*'*50}\n[{q_id}] {topic}\n{'*'*50}")

        try:
            context = recuperar_contexto(topic, topic=topic) or "N/A"
        except Exception as e:
            print(f"Error RAG: {e}")
            context = "Error retrieving context."

        resultado, historial, t_total = lanzar_debate(
            topic, context, n_rounds=N_ROUNDS
        )

        guardar_resultado(
            q_id, topic, context, historial,
            resultado, t_total, resultados_dir
        )

        time.sleep(1)

if __name__ == "__main__":
    main()