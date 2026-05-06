import requests
import re

def call_ollama(model_id, messages, max_tokens=500): # Cambiado el default a 500
    is_r1 = "deepseek-r1" in model_id
    url = "http://localhost:11434/api/chat"
    
    """
    Esta función es el corazón del sistema. 
    Carga el modelo en l 3 P40, genera y lo descarga.
    """
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "options": {
            # Si es R1, le damos 1200 para que quepa el pensamiento + 500 de respuesta
            "num_predict": 1200 if is_r1 else max_tokens, 
            "temperature": 0.7,
             #"num_ctx": 8192,
             #"num_thread": 16,
            "num_ctx": 2048,   # <--- CAMBIO: He bajado de 8192 a 2048 para mis 2GB de VRAM
            "num_thread": 4,   # <--- CAMBIO: Ajustado de 16 a 4 para un portátil normal
            "stop": ["Moderator:", "DLB:", "PNM:", "###", "Topic:"],
            "include_thoughts": False
        },
        "keep_alive": 0 # FORZAR DESCARGA INMEDIATA DE VRAM
    }

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        content = data['message']['content']

        # Limpieza de seguridad para DeepSeek-R1
        if is_r1:
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # Si el modelo devolvió algo vacío tras la limpieza
        if not content:
            content = "I need to re-evaluate the technical constraints to provide a concise answer."
        
        metrics = {
            "total_duration": data.get("total_duration", 0) / 1e9,
            "load_duration": data.get("load_duration", 0) / 1e9,
            "prompt_duration": data.get("prompt_eval_duration", 0) / 1e9,
            "eval_duration": data.get("eval_duration", 0) / 1e9, # Inferencia pura
            "token_count": data.get("eval_count", 0), # Tokens generados
            "tps": data.get("eval_count", 0) / (data.get("eval_duration", 1) / 1e9)
        }
        
        return {"content": content.strip(), "metrics": metrics}
    except Exception as e:
        return {"content": f"Error en Ollama: {str(e)}", "metrics": None}