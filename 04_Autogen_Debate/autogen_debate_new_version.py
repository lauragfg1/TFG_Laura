import autogen
import sys
import os
import json
import csv
import re
import time

sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '03_Langgraph_Parallel')
))
from qdrant_rag import recuperar_contexto

BASE_URL = "http://localhost:11434/v1"
N_ROUNDS = 5  # Ciclos argumentativos DLB+PNM equivalentes a LangGraph

def make_llm_config(model: str) -> dict:
    return {
        "config_list": [{
            "model": model,
            "api_key": "NotRequired",
            "base_url": BASE_URL
        }],
        "temperature": 0.7,
        "cache_seed": None
    }

def create_autogen_debate(topic: str, context: str, n_rounds: int = N_ROUNDS):
    """
    Estructura equivalente a LangGraph:
    - 1 Moderador (llama3.1:8b)
    - 1 DLB Expert (qwen2.5:7b) 
    - 1 PNM Expert (mistral:7b)
    - 1 Juez externo al canal (llama3.1:8b)
    
    n_rounds ciclos: Mod -> DLB -> PNM -> Mod -> DLB -> PNM ...
    Con round_robin sobre [Mod, DLB, PNM] = 3 agentes * n_rounds + 1 proxy = max_round
    """
    
    # Agentes
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message="Human admin. Provide the initial topic and context.",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    moderator = autogen.AssistantAgent(
        name="Moderator",
        system_message=(
            f"You are a Moderator in a technical debate.\n"
            f"Topic: {topic}\n\n"
            "YOUR ROLE:\n"
            "- Drive the discussion forward.\n"
            "- Prevent repetition.\n"
            "- Push for deeper understanding.\n"
            "- Maintain technical precision and neutral tone.\n"
            "Max 100 words per intervention."
        ),
        llm_config=make_llm_config("llama3.1:8b"),
    )

    dlb_expert = autogen.AssistantAgent(
        name="DLB_Expert",
        system_message=(
            f"Topic: {topic}\n"
            f"Reference Context: {context}\n\n"
            "RULES:\n"
            "1. Focus ONLY on physical mechanism and link budget implications.\n"
            "2. Explain array factor, interference, spacing effects.\n"
            "3. DO NOT invent metrics not in the Reference Context.\n"
            "4. Provide a NEW insight each round. Never repeat yourself.\n"
            "Max 100 words."
        ),
        llm_config=make_llm_config("qwen2.5:7b"),
    )

    pnm_expert = autogen.AssistantAgent(
        name="PNM_Expert",
        system_message=(
            f"Topic: {topic}\n"
            f"Reference Context: {context}\n\n"
            "RULES:\n"
            "1. Focus ONLY on payload and network management implications.\n"
            "2. Describe cause-effect relationships (e.g., more interference → lower capacity).\n"
            "3. DO NOT use numbers unless explicitly in the Reference Context.\n"
            "4. Provide a NEW insight each round. Never repeat yourself.\n"
            "Max 100 words."
        ),
        llm_config=make_llm_config("mistral:7b"),
    )

    judge = autogen.AssistantAgent(
        name="Judge",
        system_message=(
            f"You produce the final technical synthesis of a debate.\n"
            f"Topic: {topic}\n"
            f"Reference Context: {context}\n\n"
            "OUTPUT FORMAT:\n"
            "Return a JSON object with EXACTLY these 4 keys:\n"
            "1. \"topic\": the topic presented.\n"
            "2. \"winner\": choose either \"DLB\" or \"PNM\" based on who had better arguments.\n"
            "3. \"consensus_response\": WRITE A NEW PARAGRAPH (max 150 words) merging the insights from the debate into a final technical conclusion. DO NOT copy this instruction.\n"
            "4. \"reference_context\": WRITE A NEW 60-WORD SUMMARY of the factual ground-truth extracted from the reference context. DO NOT copy this instruction.\n\n"
            "Respond ONLY with the JSON dictionary. Do not include any formatting like ```json or introductory text."
        ),
        llm_config=make_llm_config("llama3.1:8b"),
    )

    # Canal: Moderador + 2 expertos, sin juez
    # 1 Admin + n_rounds * 3 agentes (Mod, DLB, PNM)
    max_round = 1 + n_rounds * 3

    groupchat = autogen.GroupChat(
        agents=[user_proxy, moderator, dlb_expert, pnm_expert],
        messages=[],
        max_round=max_round,
        speaker_selection_method="round_robin"
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=make_llm_config("llama3.1:8b")
    )

    initial_message = (
        f"Topic: {topic}\n"
        f"Context: {context}\n\n"
        "Introduce the technical problem and invite discussion. Max 100 words."
    )

    # Medir tiempo real
    t_start = time.perf_counter()
    
    chat_result = user_proxy.initiate_chat(
        manager,
        message=initial_message,
        summary_method="last_msg"
    )
    
    t_debate = time.perf_counter() - t_start

    # Juez externo — lee historial en frío
    t_judge_start = time.perf_counter()
    
    history_str = "\n".join([
        f"[{m.get('name', m.get('role', 'Unknown'))}]: {m.get('content', '')}"
        for m in chat_result.chat_history
    ])

    judge_msg = (
        f"Read the following debate and produce the JSON decision.\n\n"
        f"DEBATE:\n{history_str}"
    )

    final_synthesis = judge.generate_reply(
        messages=[{"role": "user", "content": judge_msg}]
    )
    
    t_judge = time.perf_counter() - t_judge_start
    t_total = t_debate + t_judge

    return chat_result, final_synthesis, t_total, t_debate

def extract_json(text: str):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    return None

def contar_tokens(text: str) -> int:
    """Aproximación: palabras / 0.75"""
    return max(1, int(len(str(text).split()) / 0.75))

def guardar_resultado(q_id, topic, context, chat_result, 
                      final_synthesis, t_total, resultados_dir):
    q_dir = os.path.join(resultados_dir, q_id)
    os.makedirs(q_dir, exist_ok=True)

    base_name = f"{q_id}_debateAG"

    # TXT con historial
    txt_path = os.path.join(q_dir, f"{base_name}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"TOPIC: {topic}\n{'='*60}\n")
        f.write(f"CONTEXT:\n{context}\n{'='*60}\n")
        f.write("DEBATE:\n")
        for msg in chat_result.chat_history:
            name = msg.get('name', msg.get('role', 'Unknown'))
            content = msg.get('content', '')
            f.write(f"[{name}]:\n{content}\n{'-'*40}\n")
        f.write(f"[Judge]:\n{final_synthesis}\n")

    # JSON decisión
    judge_data = extract_json(final_synthesis)
    winner = "Unknown"
    if judge_data:
        json_path = os.path.join(q_dir, f"{base_name}_decision.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(judge_data, f, indent=4)
        winner = judge_data.get("winner", "Unknown")

    # Métricas — tokens aproximados
    tokens_in = 0
    tokens_out = 0
    base_ctx = contar_tokens(context) + contar_tokens(topic)

    for i, msg in enumerate(chat_result.chat_history):
        content = msg.get('content', '')
        role = msg.get('name', msg.get('role', ''))
        tokens_out += contar_tokens(content)
        # Cada agente recibe el historial acumulado como entrada
        tokens_in += base_ctx + sum(
            contar_tokens(m.get('content', ''))
            for m in chat_result.chat_history[:i]
        )

    # Añadir tokens del juez
    tokens_in += contar_tokens(history_str) if 'history_str' in dir() else 0
    tokens_out += contar_tokens(final_synthesis)

    avg_tps = tokens_out / t_total if t_total > 0 else 0

    csv_path = os.path.join(resultados_dir, "metricas_autogen.csv")
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

        t_start = time.perf_counter()
        chat_result, final_synthesis, t_total, t_debate = create_autogen_debate(
            topic, context, n_rounds=N_ROUNDS
        )
        
        guardar_resultado(
            q_id, topic, context, chat_result,
            final_synthesis, t_total, resultados_dir
        )
        
        time.sleep(1)

if __name__ == "__main__":
    main()