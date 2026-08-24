import requests
import re
import os
import time
from llm_config import NUM_CTX, NUM_PREDICT, TEMPERATURE, STOP_SEQUENCES, REPEAT_PENALTY

# Reintentos ante errores transitorios del servidor (probado en la práctica:
# un 500 Internal Server Error puntual, no reproducible con la misma llamada
# repetida a los pocos segundos, probablemente por carga puntual en la máquina
# compartida). Sin esto, un fallo puntual en una campaña de varios días sin
# supervisión se guarda como "Error en Ollama" con métricas a cero y no se
# recupera solo.
MAX_RETRIES = 2
RETRY_WAIT_S = 5

def call_ollama(model_id, messages, force_json=False, max_tokens=500):
    # 1. LEEMOS EL CLIMA (Variables de entorno)
    size = os.environ.get("SIZE", "70B")
    mode = os.environ.get("MODE", "heterogeneo")
    
    is_r1 = "deepseek-r1" in model_id
    url = "http://localhost:11434/api/chat"
    
    # 2. CONFIGURACIÓN DINÁMICA SEGÚN EL TAMAÑO
    if size == "70B":
        request_timeout = 900
        threads = 16
    else:
        request_timeout = 300
        threads = 8

    # 3. LÓGICA DE KEEP_ALIVE
    if size == "70B" and mode == "heterogeneo":
        keep_alive_val = 0
    else:
        keep_alive_val = -1

    formatted_messages = []
    for m in messages:
        # Detectamos el rol según el tipo de objeto 
        role = "user"
        if "AI" in type(m).__name__: role = "assistant"
        elif "System" in type(m).__name__: role = "system"
        
        # Extraemos el contenido 
        content = m.content if hasattr(m, "content") else str(m)
        formatted_messages.append({"role": role, "content": content})

    # Mismo presupuesto de generación y temperatura para todos los modelos,
    # incluidos los DeepSeek-R1 (antes recibían 4x num_predict y más temperatura
    # que sus compañeros de debate, lo cual sesgaba la comparación entre condiciones).
    payload = {
        "model": model_id,
        "messages": formatted_messages,
        "stream": False,
        "options": {
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_thread": threads,
            "stop": STOP_SEQUENCES,
            "repeat_penalty": REPEAT_PENALTY,
            "include_thoughts": False
        },
        "keep_alive": keep_alive_val
    }
    
    if force_json:
        payload["format"] = "json"

    last_error = None
    for attempt in range(1, MAX_RETRIES + 2):  # intento inicial + MAX_RETRIES reintentos
        try:
            response = requests.post(url, json=payload, timeout=request_timeout)
            response.raise_for_status()
            data = response.json()
            content = data['message']['content']

            if is_r1:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

            if not content:
                content = "I need to re-evaluate the technical constraints to provide a concise answer."

            # --- MÉTRICAS
            # Conversión de nanosegundos a segundos
            t_total = data.get("total_duration", 0) / 1e9         # Tiempo total de la petición
            t_load = data.get("load_duration", 0) / 1e9           # Tiempo de carga en VRAM
            t_prompt_eval = data.get("prompt_eval_duration", 0) / 1e9 # Tiempo de procesamiento del prompt
            t_eval = data.get("eval_duration", 0) / 1e9           # Tiempo de generación de respuesta

            # Hueco entre el total que reporta Ollama y la suma de sus sub-duraciones
            # (cola/red/overhead del lado cliente), no explicado por carga+prompt+generación.
            t_overhead = max(t_total - (t_load + t_prompt_eval + t_eval), 0.0)

            # Evitar división por cero
            safe_t_eval = t_eval if t_eval > 0 else 0.001

            metrics = {
                "modelo": model_id,                               # ID del modelo utilizado
                "tiempo_total_s": round(t_total, 3),              # Total de la operación (seg)
                "tiempo_carga_s": round(t_load, 3),               # Carga del modelo (seg)
                "tiempo_prompt_eval_s": round(t_prompt_eval, 3),  # Procesamiento de entrada (seg)
                "tiempo_generacion_s": round(t_eval, 3),          # Generación de salida (seg)
                "tiempo_overhead_s": round(t_overhead, 3),        # Hueco no explicado por carga+prompt+generación
                "tokens_prompt": data.get("prompt_eval_count", 0),# Cantidad de tokens de entrada
                "tokens_generacion": data.get("eval_count", 0),   # Cantidad de tokens generados
                "tps": round(data.get("eval_count", 0) / safe_t_eval, 2) if safe_t_eval > 0 else 0, # Velocidad (tokens/seg)
                "num_predict_usado": payload["options"]["num_predict"],  # Trazabilidad del presupuesto de generación real
                "temperature_usada": payload["options"]["temperature"],  # Trazabilidad de la temperatura real
                "reintentos": attempt - 1,                        # 0 si funcionó a la primera
            }

            return {"content": content.strip(), "metrics": metrics}
        except Exception as e:
            last_error = e
            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_WAIT_S)

    return {"content": f"Error en Ollama tras {MAX_RETRIES + 1} intentos: {str(last_error)}", "metrics": None}