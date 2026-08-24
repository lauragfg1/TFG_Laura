"""Parámetros de generación compartidos por los tres frameworks (LangGraph,
AutoGen, CrewAI). Antes cada framework fijaba su propia temperatura/num_predict
por separado (LangGraph 0.5, AutoGen/CrewAI 0.7), lo que hacía que el mismo
trío de modelos se comparara bajo un muestreo distinto según el framework.
Ahora los tres importan estas constantes en vez de duplicarlas.
"""

NUM_CTX = 8192
NUM_PREDICT = 1024
TEMPERATURE = 0.5
STOP_SEQUENCES = ["Moderator:", "DLB:", "PNM:", "Topic:"]

# Se probó subir esto (1.3 y 1.15) para reducir la repetición verbatim entre
# rondas observada en pruebas manuales. Se descartó: 1.3 hizo que qwen2.5:7b
# cambiara a chino a mitad de respuesta; 1.15 produjo una respuesta vacía y
# que el moderador se inventara turnos falsos de DLB/PNM dentro de su propio
# mensaje. Con una sola muestra por valor no hay evidencia sólida de causalidad,
# pero tampoco de que valga la pena el riesgo en una campaña de 33 ejecuciones.
# Se deja el default de Ollama (1.1); la repetición queda como limitación
# documentada en vez de un parámetro ajustado a ciegas.
REPEAT_PENALTY = 1.1
