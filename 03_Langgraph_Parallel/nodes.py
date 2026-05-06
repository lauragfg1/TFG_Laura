from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from state import DebateState
from ollama_client import call_ollama  
from qdrant_rag import recuperar_contexto 
import os
import logging 

def get_models():
    size = os.environ.get("SIZE", "70B")
    mode = os.environ.get("MODE", "heterogeneo")

    if size == "2B":
        if mode == "homogeneo":
            return "gemma2:2b", "gemma2:2b", "gemma2:2b"
        return "gemma2:2b", "qwen2.5:1.5b", "deepseek-r1:1.5b"
    elif size == "8B":
        if mode == "homogeneo":
            return "llama3.1:8b", "llama3.1:8b", "llama3.1:8b"
        return "llama3.1:8b", "qwen2.5:7b", "mistral:7b"
    else:  # 70B
        if mode == "homogeneo":
            return "llama3.3:70b", "llama3.3:70b", "llama3.3:70b"
        return "llama3.3:70b", "deepseek-r1:70b", "qwen2.5:72b"

# --- NODO MODERADOR 
def moderator_node(state: DebateState):
    MODEL_MOD, _, _ = get_models()

    current_round = state["round"]
    messages = state["messages"]
    history = messages[-6:] if len(messages) > 0 else [] # Ventana deslizante para no saturar el contexto

    logging.info(f"🔄 INICIANDO RONDA: {current_round} | Moderador: {MODEL_MOD}")

    system_prompt = SystemMessage(
        content=(f"""
    You are a Moderator in a Technical debate for: {state['topic']}.

    YOUR ROLE:
    - Drive the discussion forward.
    - Prevent repetition.
    - Push for deeper understanding.
    - Maintain technical precision and neutral tone.
    """)
    )

    # Optimizamos RAG: Solo se ejecuta en la ronda 0
    if current_round == 0:
        # Introducción con RAG inicial
        context = recuperar_contexto(state["topic"], topic=state["topic"])
        user_prompt = (
            f"Context: {context}\n\n"
            "Introduce the technical problem clearly and invite discussion from both experts. Max 100 words."
        )
    else:
        # Recuperamos el contexto ya guardado para no repetir la búsqueda
        context = state.get("rag_context", "N/A")
        # Recuperar lo último de cada experto
        last_dlb = next((m.content for m in reversed(messages) if "[DLB]" in m.content), "N/A")
        last_pnm = next((m.content for m in reversed(messages) if "[PNM]" in m.content), "N/A")

        user_prompt = (
            f"DLB Analysis: {last_dlb}\n\n"
            f"PNM Analysis: {last_pnm}\n\n"

            "TASKS:"
            "-Summarize key points and introduce a new technical aspect."
            "-Prevent repetition."
            "-Push for deeper reasoning. Max 100 words."
        )

    # Combinamos el historial + instrucción de sistema + prompt actual
    full_prompt_list = history + [system_prompt, HumanMessage(content=user_prompt)]
    
    result = call_ollama(MODEL_MOD, full_prompt_list)
    
    return {
        "round": current_round + 1,
        "messages": [AIMessage(content=f"[MODERATOR] {result['content']}")],
        "rag_context": context, 
        "metrics": [result["metrics"]] if result.get("metrics") else []
    }

# ------ DLB Expert Node (Link Budget) 
def dlb_node(state: DebateState):
    _, MODEL_DLB, _ = get_models()

    logging.info(f"📊 [DLB] Generando análisis técnico con {MODEL_DLB}...")
    
    last_mod_msg = state["messages"][-1].content
    history = state["messages"][-6:]
    
    # Reutilizamos el contexto del estado en lugar de llamar a recuperar_contexto
    context = state.get("rag_context", "No context available.")

    prompt = f"""
        Your answers focuses on the **budget** aspect of the design and link. You do not care any other aspect.

        Topic: {state['topic']}
        Moderator said: {last_mod_msg}
        Context: {context}

        RULES:
        1. Explain the PHYSICAL mechanism (array factor, interference, spacing effects).
        2. Do NOT repeat generic textbook statements. Provide detailed, specific analysis based on the Context.
        3. Focus on the budget implications of the design and link. Not budget in terms of money, but in terms of "what is the cost in terms of performance, capacity, interference, etc." of the design choices.
        4. DO NOT invent or fabricate any metric, number, or system fact that is not stated in the Context. If the Context lacks data, explicitly state so, and rely strictly on universal physics principles to deduce the answer.
        5. Don´t repeat what you said in previous rounds. Always provide a NEW technical insight or angle of the problem.

        Be highly technical and concise. Max 100 words.
        """

    result = call_ollama(MODEL_DLB, history + [HumanMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=f"[DLB] {result['content']}")],
        "metrics": [result["metrics"]] if result.get("metrics") else []
    }

# ------ PNM Expert Node (Payload & Network)
def pnm_node(state: DebateState):
    _, _, MODEL_PNM = get_models()

    logging.info(f"📡 [PNM] Calculando recursos de red con {MODEL_PNM}...")
    
    last_mod_msg = state["messages"][-1].content
    history = state["messages"][-6:]
    
    # Reutilizamos el contexto del estado en lugar de llamar a recuperar_contexto
    context = state.get("rag_context", "No context available.")
    
    prompt = f"""
        Your answers focuses on the Payload and Network Management aspect. You do not care any other aspect.

        Topic: {state['topic']}
        Moderator said: {last_mod_msg}
        Context: {context}

        RULES:
        1. Focus on system/network implications only.
        2. Describe relationships (e.g., "more interference → lower capacity").
        3. DO NOT use numbers, percentages, or metrics unless they are EXPLICITLY contained in the Reference Context.
        4. Provide specific technical deductions. Avoid vague generalizations. If the Context lacks specifics, base your reasoning purely on universal networking/telecommunications principles without making up data.
        5. Don´t repeat what you said in previous rounds. Always provide a NEW technical insight of the problem.

        Be highly technical and concise. Max 100 words. 
        """

    result = call_ollama(MODEL_PNM, history + [HumanMessage(content=prompt)])

    return {
        "messages": [AIMessage(content=f"[PNM] {result['content']}")],
        "metrics": [result["metrics"]] if result.get("metrics") else []
    }

#---------- ROUTER (Control de flujo) 
def debate_router(state: DebateState):
    if state["round"] < state["max_rounds"]:
        return ["dlb", "pnm"] 
    return ["mod"]

# ---------  LLM JUDGE NODE 

def final_answer_node(state: DebateState):
    MODEL_MOD, _, _ = get_models()

    logging.info(f"⚖️ [MODERADOR] Generando respuesta consenso con {MODEL_MOD}...")
    
    # El juez analiza todo el historial para decidir
    history = "\n".join([m.content for m in state["messages"]])
    context = state.get("rag_context", "No context available.")

    prompt = f"""You are the FINAL SYNTHESIS for a technical debate of this topic: "{state['topic']}".

    HISTORY:
    {history}

    YOUR ROLE:
    - Produce the FINAL TECHNICAL DECISION based on the debate.
    - Merge DLB (physical layer) and PNM (system impact) conclusions into ONE coherent explanation.
    - The first sentence MUST directly answer the topic question. "{state['topic']}"
    - If the history does not explicitly provide numbers or specific data, DO NOT fabricate them. 

    REFERENCE CONTEXT:
    {context}

    OUTPUT FORMAT:
    Return ONLY VALID JSON. Do not write anything outside the JSON structure.

    {{
        "topic": "{state['topic']}",
        "winner": "PNM",
        "consensus_response": "Write your deep, technical, cause-effect explanation here merging DLB and PNM insights. Max 150 words.",
        "reference_context": "Write a 60-word summary of the factual ground-truth extracted from the reference context provided above. Do NOT copy the expert's chat history."
    }}

    """

    result = call_ollama(MODEL_MOD, [HumanMessage(content=prompt)], force_json=True)

    return {
        "messages": [AIMessage(content=f"FINAL DECISION:\n{result['content']}")],
        "metrics": [result["metrics"]] if result.get("metrics") else []
    }