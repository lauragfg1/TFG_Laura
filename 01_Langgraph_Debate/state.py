import operator
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph.message import add_messages

class DebateState(TypedDict):
    """
    Representa el estado compartido (Shared State) dentro del grafo de LangGraph.
    """
    # Historial completo (se acumula sin límite a propósito: hace falta íntegro para el
    # transcript .txt y para sumar métricas). Los nodos NUNCA lo pasan tal cual al LLM:
    # siempre lo acotan primero (ver RECENT_WINDOW/FINAL_SYNTHESIS_WINDOW en nodes.py).
    messages: Annotated[List, add_messages]  # Anexa los nuevos mensajes al historial existente

    # Control del debate
    round: int                    
    max_rounds: int                         
    topic: str
    rag_context: str
    
    # Métricas
    metrics: Annotated[List[dict], operator.add]