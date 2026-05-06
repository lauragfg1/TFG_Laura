import operator
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph.message import add_messages

class DebateState(TypedDict):
    """
    Representa el estado compartido (Shared State) dentro del grafo de LangGraph.
    """
    # Historial
    messages: Annotated[List, add_messages]  # Anexa los nuevos mensajes al historial existente 

    # Control del debate
    round: int                    
    max_rounds: int                         
    topic: str
    rag_context: str
    
    # Métricas
    metrics: Annotated[List[dict], operator.add]