import autogen
import sys
import os
import json
import csv
import re
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '03_Langgraph_Parallel')))
from qdrant_rag import recuperar_contexto

# Configuración de los modelos locales usando la API compatible con OpenAI de Ollama.
# Para asignar un modelo diferente a cada agente simulando el modo "heterogeneo" de LangGraph:

base_url = "http://localhost:11434/v1"

llm_config_mod = {
    "config_list": [{"model": "llama3.1:8b", "api_key": "NotRequired", "base_url": base_url}],
    "temperature": 0.7, "cache_seed": None
}

llm_config_dlb = {
    "config_list": [{"model": "qwen2.5:7b", "api_key": "NotRequired", "base_url": base_url}],
    "temperature": 0.7, "cache_seed": None
}

llm_config_pnm = {
    "config_list": [{"model": "mistral:7b", "api_key": "NotRequired", "base_url": base_url}],
    "temperature": 0.7, "cache_seed": None
}

def create_autogen_debate(topic: str, context: str, max_rounds: int = 5):
    # 1. Agente Proxy (Simula al usuario/sistema que inicia el debate y provee el contexto inicial)
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message="A human admin. Provide the initial context and topic to the moderator.",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    # 2. Moderador
    moderator = autogen.AssistantAgent(
        name="Moderator",
        system_message=(
            f"You are a Moderator in a Technical debate for: {topic}.\n\n"
            "YOUR ROLE:\n"
            "- Drive the discussion forward.\n"
            "- Prevent repetition.\n"
            "- Push for deeper understanding.\n"
            "- Maintain technical precision and neutral tone."
        ),
        llm_config=llm_config_mod,
    )

    # 3. Experto DLB
    dlb_expert = autogen.AssistantAgent(
        name="DLB_Expert",
        system_message=(
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
        llm_config=llm_config_dlb,
    )

    # 4. Experto PNM
    pnm_expert = autogen.AssistantAgent(
        name="PNM_Expert",
        system_message=(
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
        llm_config=llm_config_pnm,
    )

    judge_prompt = (
        f"You are the FINAL SYNTHESIS for a technical debate of this topic: \"{topic}\".\n\n"
        "YOUR ROLE:\n"
        "- Produce the FINAL TECHNICAL DECISION based on the debate.\n"
        "- Merge DLB (physical layer) and PNM (system impact) conclusions into ONE coherent explanation.\n"
        f"- The first sentence MUST directly answer the topic question. \"{topic}\"\n"
        "- If the history does not explicitly provide numbers or specific data, DO NOT fabricate them. \n\n"
        f"REFERENCE CONTEXT:\n{context}\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY VALID JSON. Do not write anything outside the JSON structure.\n\n"
        "{\n"
        f"    \"topic\": \"{topic}\",\n"
        "    \"winner\": \"PNM\",\n"
        "    \"consensus_response\": \"Write your deep, technical, cause-effect explanation here merging DLB and PNM insights. Max 150 words.\",\n"
        "    \"reference_context\": \"Write a 60-word summary of the factual ground-truth extracted from the reference context provided above. Do NOT copy the expert's chat history.\"\n"
        "}"
    )

    # 5. Juez / Sintetizador (Fuera del GroupChat)
    judge = autogen.AssistantAgent(
        name="Judge",
        system_message=judge_prompt,
        llm_config=llm_config_mod,
    )

    # Canal de debate: solo perfiles argumentativos
    groupchat = autogen.GroupChat(
        agents=[user_proxy, moderator, dlb_expert, pnm_expert],
        messages=[],
        max_round=max_rounds,
        speaker_selection_method="round_robin"
    )

    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config_mod # El GroupChatManager suele requerir un modelo capaz como "orquestador"
    )

    # 7. Iniciar la conversación
    initial_message = (
        f"Context: {context}\n\n"
        "Introduce the technical problem clearly and invite discussion from both experts. Max 100 words."
    )
    
    print("\n" + "="*50)
    print("🚀 INICIANDO DEBATE AUTOGEN CON OLLAMA")
    print("="*50 + "\n")
    
    # 8. Arrancar el chat entre los expertos y proxy
    chat_result = user_proxy.initiate_chat(
        manager,
        message=initial_message,
        summary_method="last_msg"
    )
    
    # 9. Síntesis asíncrona "en frío" (Aislamiento cognitivo)
    print("⚖️ Generando evaluación del Juez...")
    history_str = ""
    for msg in chat_result.chat_history:
        name = msg.get('name', msg.get('role', 'Unknown'))
        content = msg.get('content', '')
        history_str += f"[{name}]: {content}\n"
        
    juez_eval_msg = f"Read the following debate and output the JSON decision.\n\nDEBATE HISTORY:\n{history_str}"
    final_judge_reply = judge.generate_reply(messages=[{"role": "user", "content": juez_eval_msg}])
    
    chat_result.summary = final_judge_reply
    
    return chat_result

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

def guardar_resultado_debate(q_id: str, topic: str, context: str, chat_result, base_resultados_dir: str, duration_s: float):
    """Guarda el historial del debate y el contexto en un archivo de texto bien formateado."""
    # Crear subcarpeta para este debate
    resultados_dir = os.path.join(base_resultados_dir, q_id)
    os.makedirs(resultados_dir, exist_ok=True)
    
    base_name = f"{q_id}_debateAG"
    nombre_archivo_salida = f"{base_name}.txt"
    ruta_salida = os.path.join(resultados_dir, nombre_archivo_salida)
    
    last_msg = ""
    try:
        with open(ruta_salida, 'w', encoding='utf-8') as f_out:
            f_out.write(f"📝 TEMA DEBATIDO: {topic}\n")
            f_out.write("="*80 + "\n\n")
            f_out.write(f"📚 CONTEXTO:\n{context}\n\n")
            f_out.write("="*80 + "\n")
            f_out.write("🗣️ HISTORIAL DEL DEBATE:\n")
            f_out.write("="*80 + "\n\n")
            
            # Iteramos sobre los mensajes del chat
            if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                for msg in chat_result.chat_history:
                    nombre = msg.get('name', msg.get('role', 'Unknown'))
                    contenido = msg.get('content', '')
                    f_out.write(f"[{nombre}]:\n{contenido}\n")
                    f_out.write("-" * 50 + "\n")
            else:
                f_out.write("No se pudo obtener el historial de PyAutogen o el debate está vacío.\n")
                
            if hasattr(chat_result, 'summary') and chat_result.summary:
                last_msg = chat_result.summary
                f_out.write("[Judge - Summary]:\n")
                f_out.write(f"{last_msg}\n")
                f_out.write("-" * 50 + "\n")
                
        print(f"\n💾 Debate guardado con éxito en: {ruta_salida}")
    except Exception as e:
        print(f"❌ Error guardando el archivo de resultados: {e}")
        
    # Guardar métricas y JSON del Juez equivalentemente a LangGraph
    # Extracción
    judge_data = extract_judge_json(last_msg)
    winner = "Unknown"
    if judge_data:
        json_path = os.path.join(resultados_dir, f"{base_name}_decision.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(judge_data, f, indent=4)
        winner = judge_data.get("winner", "Unknown")

    # En AutoGen con custom models a veces los usage caen en otra estructura si no es OAI real.
    # Vamos a contarlos sumando la info de la historia que guardamos si cost no funciona.
    tokens_out = 0
    base_input_tokens = max(1, int((len(context.split()) + len(topic.split())) / 0.75))
    
    if hasattr(chat_result, 'chat_history'):
        # Multiplicamos el contexto inicial por los turnos, emulando la acumulación
        turnos = len(chat_result.chat_history)
        tokens_in = base_input_tokens * max(1, turnos)
        for msg in chat_result.chat_history:
            content = msg.get('content', '')
            if content:
                # Aproximacion rapida de tokens out (palabras / 0.75)
                if msg.get('role') == 'assistant' or msg.get('name') != 'Admin':
                    tokens_out += max(1, int(len(content.split()) / 0.75))
    else:
        tokens_in = 0
    
    # Intento de API OAI real (fallback)
    if hasattr(chat_result, 'cost') and isinstance(chat_result.cost, dict):
        usage = chat_result.cost.get('usage_excluding_cached_inference', {})
        for key, costs in usage.items():
            if isinstance(costs, dict) and costs.get('completion_tokens', 0) > 1: # si nos devuelve 1 es falso
                tokens_out = costs.get('completion_tokens', 0)
                tokens_in = costs.get('prompt_tokens', 0)
    
    # Algunas métricas (como load, prompt_s) no se devuelven por default vía API OpenAI sin hacks.
    # Pondremos lo esencial para el CSV.
    csv_path = os.path.join(base_resultados_dir, "metricas_autogen.csv")
    file_exists = os.path.exists(csv_path)
    
    avg_tps = (tokens_out / duration_s) if duration_s > 0 else 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Misma estructura que LangGraph
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
    # Directorios
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
    
    # Procesar 50 primeros
    for idx, topic in enumerate(preguntas[:50]):
        q_id = f"q{idx + 1:03d}"
        
        print("\n" + "*"*60)
        print(f"📖 [{q_id}] Procesando pregunta...")
        print(f"🗣️ TOPIC: {topic}")
        print("*"*60)
        
        # 1. Recuperar Contexto (RAG)
        print("🔍 Recuperando contexto de Qdrant...")
        try:
            context = recuperar_contexto(topic, topic=topic)
            context = context if context else "N/A"
            if context != "N/A": 
                print(f"✅ Contexto recuperado: {context[:100]}...\n")
        except Exception as e:
            print(f"❌ Error RAG: {e}")
            context = "Error in retrieving context."
            
        # 2. Arrancar Autogen
        start_t = time.time()
        # Modificado para igualar las iteraciones de LangGraph (5 ciclos). 
        # En AutoGen cada mensaje cuenta como 1 round. 
        # 1 proxy + 5 ciclos * 3 agentes (mod, dlb, pnm) = 16
        chat_result = create_autogen_debate(topic, context, max_rounds=16)
        end_t = time.time()
        
        # 3. Guardar resultados
        guardar_resultado_debate(q_id, topic, context, chat_result, resultados_dir, end_t - start_t)

if __name__ == "__main__":
    main()
