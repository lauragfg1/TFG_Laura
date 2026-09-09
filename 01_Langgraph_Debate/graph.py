from langgraph.graph import StateGraph, END
from state import DebateState
from nodes import moderator_node, dlb_node, pnm_node, final_answer_node, debate_router

# Instanciamos el grafo con el estado que tiene los Reducers (add_messages)
workflow = StateGraph(DebateState)

# Definición de Nodos
workflow.add_node("moderator", moderator_node)
workflow.add_node("dlb", dlb_node)
workflow.add_node("pnm", pnm_node)
workflow.add_node("moderator_final", final_answer_node)

# Punto de entrada
workflow.set_entry_point("moderator")

# Depende de lo que el router devuelva "dlb", "pnm" o "juez", activa este nodo
workflow.add_conditional_edges(
    "moderator",
    debate_router,
    {
        "dlb": "dlb",  
        "pnm": "pnm",  
        "mod": "moderator_final"  
    }
)

# AMBOS nodos deben tener una arista de vuelta al moderador.
# LangGraph esperará a que ambos terminen antes de volver al moderador.
workflow.add_edge("dlb", "moderator")
workflow.add_edge("pnm", "moderator")

# Finalización
workflow.add_edge("moderator_final", END)

# Compilación
app = workflow.compile()

print("🚀 Grafo compilado con éxito. Sistema paralelo listo.")

# --- Generación de la imagen del grafo ---
try:
    with open("grafo_tfg.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    print("🖼️ Imagen del grafo guardada como 'grafo_tfg.png'")
except Exception:
    print("⚠️ No se pudo generar el PNG automáticamente.")