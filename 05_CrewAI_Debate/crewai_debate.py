import os
import sys
import time
import json
import csv
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03_Langgraph_Parallel')))
from qdrant_rag import recuperar_contexto

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DE MODELOS
# ---------------------------------------------------------
MODEL_MOD = "llama3.1:8b"
MODEL_DLB = "qwen2.5:7b"
MODEL_PNM = "mistral:7b"
BASE_URL = "http://localhost:11434/v1"

# Instanciamos los LLMs
llm_mod = ChatOpenAI(model=MODEL_MOD, base_url=BASE_URL, api_key="NotRequired", temperature=0.7)
llm_dlb = ChatOpenAI(model=MODEL_DLB, base_url=BASE_URL, api_key="NotRequired", temperature=0.7)
llm_pnm = ChatOpenAI(model=MODEL_PNM, base_url=BASE_URL, api_key="NotRequired", temperature=0.7)

# ---------------------------------------------------------
# 2. DEFINICIÓN DE AGENTES
# ---------------------------------------------------------
def crear_agentes(topic, context):
    moderador = Agent(
        role="Moderator",
        goal="Drive the discussion forward, prevent repetition, push for deeper understanding.",
        backstory=(
            f"You are a Moderator in a Technical debate for: {topic}.\n\n"
            "YOUR ROLE:\n"
            "- Drive the discussion forward.\n"
            "- Prevent repetition.\n"
            "- Push for deeper understanding.\n"
            "- Maintain technical precision and neutral tone."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm_mod
    )

    experto_dlb = Agent(
        role="DLB_Expert",
        goal="Explain the PHYSICAL mechanism and focus on the budget implications of the design choices.",
        backstory=(
            f"Your answers focuses on the **budget** aspect of the design and link. You do not care any other aspect.\n\n"
            f"Topic: {topic}\n"
            f"Context: {context}\n\n"
            "RULES:\n"
            "1. Explain the PHYSICAL mechanism (array factor, interference, spacing effects).\n"
            "2. Do NOT repeat generic textbook statements. Provide detailed, specific analysis based on the Context.\n"
            "3. Focus on the budget implications of the design and link. Not budget in terms of money, but in terms of \"what is the cost in terms of performance, capacity, interference, etc.\" of the design choices.\n"
            "4. DO NOT invent or fabricate any metric, number, or system fact that is not stated in the Context. If the Context lacks data, explicitly state so, and rely strictly on universal physics principles to deduce the answer.\n"
            "5. Don´t repeat what you said in previous rounds. Always provide a NEW technical insight or angle of the problem.\n\n"
            "Be highly technical and concise. Max 100 words."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm_dlb
    )

    experto_pnm = Agent(
        role="PNM_Expert",
        goal="Focus on the Payload and Network Management aspect and system/network implications.",
        backstory=(
            f"Your answers focuses on the Payload and Network Management aspect. You do not care any other aspect.\n\n"
            f"Topic: {topic}\n"
            f"Context: {context}\n\n"
            "RULES:\n"
            "1. Focus on system/network implications only.\n"
            "2. Describe relationships (e.g., \"more interference → lower capacity\").\n"
            "3. DO NOT use numbers, percentages, or metrics unless they are EXPLICITLY contained in the Reference Context.\n"
            "4. Provide specific technical deductions. Avoid vague generalizations. If the Context lacks specifics, base your reasoning purely on universal networking/telecommunications principles without making up data.\n"
            "5. Don´t repeat what you said in previous rounds. Always provide a NEW technical insight of the problem.\n\n"
            "Be highly technical and concise. Max 100 words."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm_pnm
    )

    juez = Agent(
        role="Judge",
        goal="Produce the FINAL TECHNICAL DECISION based on the debate.",
        backstory=(
            f"You are the FINAL SYNTHESIS for a technical debate of this topic: \"{topic}\".\n\n"
            "YOUR ROLE:\n"
            "- Produce the FINAL TECHNICAL DECISION based ONLY on the debate.\n"
            "- Merge DLB and PNM conclusions into ONE coherent explanation.\n"
            f"REFERENCE CONTEXT:\n{context}\n\n"
            "OUTPUT FORMAT:\n"
            "Return a JSON object with EXACTLY these 4 keys:\n"
            "1. \"topic\": the topic presented.\n"
            "2. \"winner\": choose either \"DLB\" or \"PNM\" based on who had better arguments.\n"
            "3. \"consensus_response\": WRITE A NEW PARAGRAPH (max 150 words) merging the insights from the debate into a final technical conclusion. DO NOT copy this instruction.\n"
            "4. \"reference_context\": WRITE A NEW 60-WORD SUMMARY of the factual ground-truth extracted from the reference context. DO NOT copy this instruction.\n\n"
            "Respond ONLY with the JSON dictionary. Do not include any formatting like ```json or introductory text."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm_mod
    )
    
    return moderador, experto_dlb, experto_pnm, juez

# ---------------------------------------------------------
# 3. DEFINICIÓN DE TAREAS Y ORQUESTACIÓN (CREW)
# ---------------------------------------------------------
def lanzar_crewai_debate(topic: str, context: str, iteraciones: int = 5):
    moderador, experto_dlb, experto_pnm, juez = crear_agentes(topic, context)

    tarea_intro = Task(
        description=f"Introduce the technical problem clearly based on the Topic: '{topic}' and Context: '{context}'. Invite discussion. Max 100 words.",
        expected_output="Introduction of the debate.",
        agent=moderador
    )
    
    tareas = [tarea_intro]
    tareas_dlb = []
    tareas_pnm = []

    for i in range(iteraciones):
        # La tarea DLB recibe por contexto la tarea anterior (intro o el último PNM)
        ctx_dlb = [tareas[-1]] if tareas else []
        t_dlb = Task(
            description=f"[ROUND {i+1}/{iteraciones}] Topic: '{topic}'. Context: '{context}'. Provide your DLB analysis. Tell us something new.",
            expected_output=f"DLB technical analysis round {i+1} (max 100 words).",
            agent=experto_dlb,
            context=ctx_dlb
        )
        
        # La tarea PNM recibe por contexto el argumento que acaba de dar DLB
        t_pnm = Task(
            description=f"[ROUND {i+1}/{iteraciones}] Topic: '{topic}'. Read DLB's latest input. Provide your PNM analysis countering or adding to it.",
            expected_output=f"PNM technical analysis round {i+1} (max 100 words).",
            agent=experto_pnm,
            context=[t_dlb]
        )
        tareas.extend([t_dlb, t_pnm])
        tareas_dlb.append(t_dlb)
        tareas_pnm.append(t_pnm)

    tarea_sintesis = Task(
        description=f"Read the entire {iteraciones}-round debate and produce the final JSON decision. Use Context: '{context}'.",
        expected_output="A strict JSON object with topic, winner, consensus_response, and reference_context.",
        agent=juez,
        context=tareas # Pasa explícitamente todo el historial al Juez
    )
    tareas.append(tarea_sintesis)

    crew = Crew(
        agents=[moderador, experto_dlb, experto_pnm, juez],
        tasks=tareas,
        verbose=True,
        process=Process.sequential 
    )

    print(f"\n🚀 Lanzando CrewAI ({iteraciones} rondas) para el topic: {topic}")
    resultado_final = crew.kickoff()
    
    # Reconstruimos el historial combinando todas las rondas
    historial = {
        "Moderator": getattr(tarea_intro.output, 'raw', str(tarea_intro.output))
    }
    
    dlb_texts = []
    pnm_texts = []
    for i in range(iteraciones):
        dlb_texts.append(f"[Round {i+1}] " + getattr(tareas_dlb[i].output, 'raw', str(tareas_dlb[i].output)))
        pnm_texts.append(f"[Round {i+1}] " + getattr(tareas_pnm[i].output, 'raw', str(tareas_pnm[i].output)))
        
    historial["DLB_Expert"] = "\n\n".join(dlb_texts)
    historial["PNM_Expert"] = "\n\n".join(pnm_texts)
    historial["Judge"] = getattr(tarea_sintesis.output, 'raw', str(tarea_sintesis.output))

    return resultado_final, historial

# ---------------------------------------------------------
# 4. GUARDADO DE RESULTADOS Y MAIN
# ---------------------------------------------------------
def extract_judge_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
    except Exception as e:
        print(f"⚠️ Error intentando reparar el JSON: {e}")
    return None

def guardar_resultado(q_id: str, topic: str, context: str, historial: dict, final_output: str, resultados_dir: str, duration_s: float):
    # Crear subcarpeta
    q_dir = os.path.join(resultados_dir, q_id)
    os.makedirs(q_dir, exist_ok=True)
    
    base_name = f"{q_id}_debateC"
    ruta_salida = os.path.join(q_dir, f"{base_name}.txt")
    json_path = os.path.join(q_dir, f"{base_name}_decision.json")
    
    try:
        with open(ruta_salida, 'w', encoding='utf-8') as f:
            f.write(f"📝 TEMA DEBATIDO: {topic}\n")
            f.write("="*80 + "\n\n")
            f.write(f"📚 CONTEXTO:\n{context}\n\n")
            f.write("="*80 + "\n")
            f.write("🗣️ HISTORIAL DE TAREAS (CREWAI Process.sequential):\n")
            f.write("="*80 + "\n\n")
            for role, text in historial.items():
                f.write(f"[{role}]:\n{text}\n")
                f.write("-" * 50 + "\n")
            f.write("⚖️ VEREDICTO FINAL (JUEZ):\n")
            f.write("="*80 + "\n")
            f.write(f"{final_output}\n")
            
        print(f"\n💾 Resultado CrewAI guardado en: {ruta_salida}")
    except Exception as e:
        print(f"❌ Error guardando fichero: {e}")

    # JSON y Métricas
    judge_data = extract_judge_json(final_output)
    winner = "Unknown"
    if judge_data:
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(judge_data, f, indent=4)
            winner = judge_data.get("winner", "Unknown")
        except Exception:
            pass

    # Aprox tokens (Contexto y topic enviado en cada turno)
    base_input_tokens = max(1, int((len(context.split()) + len(topic.split())) / 0.75))
    tokens_in = base_input_tokens * max(1, len(historial))
    tokens_out = 0
    
    for role, text in historial.items():
        if text:
            tokens_out += max(1, int(len(text.split()) / 0.75))
            
    avg_tps = (tokens_out / duration_s) if duration_s > 0 else 0

    csv_path = os.path.join(resultados_dir, "metricas_crewai.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "ID", "Size", "Mode", "Winner", "Total_S", "Load_S", "Prompt_S", 
                "Gen_S", "Tokens_In", "Tokens_Out", "Avg_TPS", "Topic"
            ])
            
        writer.writerow([
            base_name, "8B", "heterogeneo", winner, 
            round(duration_s, 2), 0.0, 0.0, 
            round(duration_s, 2), tokens_in, tokens_out, 
            round(avg_tps, 2), topic
        ])


def main():
    resultados_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Resultados'))
    dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset_Preguntas'))
    os.makedirs(resultados_dir, exist_ok=True)
    
    if not os.path.exists(dataset_dir):
        print(f"❌ No se encuentra la carpeta: {dataset_dir}")
        sys.exit(1)
        
    index_path = os.path.join(dataset_dir, 'index.txt')
    if not os.path.exists(index_path):
        print(f"❌ No se encuentra: {index_path}")
        sys.exit(1)
        
    with open(index_path, 'r', encoding='utf-8') as f:
        preguntas = [line.strip() for line in f.readlines() if line.strip()]
    
    if not preguntas:
        print("❌ index.txt está vacío.")
        sys.exit(1)
        
    print(f"📁 Cargadas {len(preguntas)} preguntas de index.txt")
    
    # Procesar 50 primeros como en Autogen
    for idx, topic in enumerate(preguntas[:50]):
        q_id = f"q{idx + 1:03d}"
        
        print("\n" + "*"*60)
        print(f"📖 [{q_id}] Procesando pregunta...")
        print(f"🗣️ TOPIC: {topic}")
        print("*"*60)
        
        print("🔍 Recuperando contexto de Qdrant...")
        try:
            context = recuperar_contexto(topic, topic=topic)
            context = context if context else "N/A"
            if context != "N/A": 
                print(f"✅ Contexto recuperado: {context[:100]}...\n")
        except Exception as e:
            print(f"❌ Error RAG: {e}")
            context = "Error in retrieving context."
            
        start_t = time.time()
        resultado_final, historial = lanzar_crewai_debate(topic, context)
        end_t = time.time()
        
        guardar_resultado(q_id, topic, context, historial, str(resultado_final), resultados_dir, end_t - start_t)

if __name__ == "__main__":
    main()