import os, requests, json, time, regex as re
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# --- Configuración de Modelos y Rutas ---
MODEL1_NAME = "llama3.2:1b"  #Moderador 
MODEL2_NAME = "qwen2.5:0.5b" #Agente DLB
MODEL3_NAME = "qwen2.5:0.5b" #Agente PNM

QDRANT_URL = "http://localhost:6333" 
OLLAMA_URL = "http://localhost:11434/api/chat"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
COLLECTION_NAME = "documents_satcom_uma"

# --- Inicialización de Motores ---
print("Conectando a Qdrant...")
qclient = QdrantClient(url=QDRANT_URL)
embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

def slugify(text):
    """Genera nombres de archivo válidos eliminando caracteres prohibidos en Windows"""
    text = text.lower()
    return re.sub(r'\W+', '_', text).strip('_')[:50]

def recuperar_contexto(query, k=5):
    """
    Motor RAG: Recupera los k fragmentos más similares 
    utilizando el método moderno de Qdrant.
    """
    try:
        # 1. Convertimos la pregunta en un vector numérico
        query_vector = embedder.encode(query).tolist()
        # 2. Buscamos directamente en la colección
        response = qclient.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=k
        )
        # 3. Extraemos el contenido técnico de cada 'hit' encontrado
        contextos = [p.payload.get('content', '') for p in response.points]
        return "\n".join(contextos)
    except Exception as e:
        print(f"⚠️ Error en la recuperación RAG: {e}")
        return "No context available."

def call_ollama(model_name, messages):
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": 500,
            "temperature": 0.7,
            "num_ctx": 2048 #Optimización para modelos pequeños
        },
        "keep_alive": 0 
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        # --- Extracción de métricas (Conversión ns a s: 1 s = 1,000,000,000 ns) ---
        load_sec      = data.get('load_duration', 0) / 1e9
        prompt_sec    = data.get('prompt_eval_duration', 0) / 1e9 
        inference_sec = data.get('eval_duration', 0) / 1e9       
        total_sec     = data.get('total_duration', 0) / 1e9   
        
        tokens_gen = data.get('eval_count', 0)
        tps = tokens_gen / inference_sec if inference_sec > 0 else 0
        
        # Reporte completo para la consola
        print(f"\n[METRICS - {model_name}]")
        print(f"  - Carga en VRAM: {load_sec:.2f} s")
        print(f"  - Procesamiento Prompt: {prompt_sec:.2f} s")
        print(f"  - Inferencia Pura: {inference_sec:.2f} s")
        print(f"  - Velocidad: {tps:.2f} tokens/s")
        print(f"  - Tiempo Total: {total_sec:.2f} s")
        
        # Limpieza radical de etiquetas de pensamiento
        full_content = data['message']['content']
        clean_content = re.sub(r'<think>.*?</think>', '', full_content, flags=re.DOTALL).strip()
        return clean_content
    except Exception as e:
        return f"Error en Ollama ({model_name}): {str(e)}"

# --- Clase Agente ----

class Agent:
    def __init__(self, name, model_id, system_prompt):
        self.name = name
        self.model_id = model_id
        self.messages = [{"role": "system", "content": system_prompt}]
    
    def generate(self, message):        
        context = recuperar_contexto(message)
        # Cada intervención se enriquece con contexto RAG
        prompt_con_contexto = f"[Context]{context}\n\n{message}"
        self.messages.append({"role": "user", "content": prompt_con_contexto})
        
        print(f"[{self.name}] Cargando modelo {self.model_id}...")
        response = call_ollama(self.model_id, self.messages)

        self.messages.append({"role": "assistant", "content": response})
        return response

# --- Clase Moderador ----
# Controla los turnos, resume argumentos y dicta la decisión final

class Moderator:
    def __init__(self, topic, model_id, max_rounds=5):
        self.topic = topic
        self.model_id = model_id
        self.max_rounds = max_rounds
        self.round = 1
        self.history = [{"role": "system", "content": (
            f"You are the moderator of a technical debate topic:  {self.topic}. "
             "Maintain technical precision and neutral tone. Max 100 words per response."
        )}]
    
    def initiate_debate(self):
        context = recuperar_contexto(self.topic)
        prompt = f"[Context]\n{context}\n\nDebate topic: {self.topic}. Start the debate."
        self.history.append({"role": "user", "content": prompt})

        response = call_ollama(self.model_id, self.history)
        self.history.append({"role": "assistant", "content": response})
        return response
    
    def analyze_and_respond(self, msg1, msg2):
        instr = f"Summarize key points and introduce a new technical aspect. DLB said: {msg1}. PNM said: {msg2}."
        self.history.append({"role": "user", "content": instr})

        response = call_ollama(self.model_id, self.history)
        self.history.append({"role": "assistant", "content": response})
        return response
    
    def final_decision(self):
        instr = "Produce the FINAL TECHNICAL DECISION based on the debate. Max 150 tokens."
        self.history.append({"role": "user", "content": instr})
        return call_ollama(self.model_id, self.history)
    
    def check_termination(self):
        res = self.round >= self.max_rounds
        self.round += 1
        return res

# --- Lógica Principal de Ejecución ---

def ejecutar_debate_completo(topic, outfile, decision_dir=None):

    # Inicialización
    moderator = Moderator(topic, model_id=MODEL1_NAME)
    dlb = Agent("DLB", MODEL2_NAME, "You are the Design and Link Budget Expert (DLB). Your answers focuses on the **budget** aspect of the design and link. You do not care any other aspect ...")
    pnm = Agent("PNM", MODEL3_NAME, "You are the Payload and Network Management Expert (PNM). Your answers focuses on the Payload and Network Management aspect. You do not care any other aspect ...")

    with open(outfile, "w", encoding="utf-8") as f:
        f.write(f"DEBATE: {topic}\n\n")
        
        # Inicio del debate
        mod_response = moderator.initiate_debate()
        f.write(f"Moderador: {mod_response}\n\n")
        print(f"\n-------- ROUND 1 ---------")
        
        dlb_response = ""
        pnm_response = ""

        while True:
            
            # Si no es la primera ronda, el moderador analiza lo anterior
            if moderator.round > 1:
                mod_response = moderator.analyze_and_respond(dlb_response, pnm_response)
                f.write(f"--- Round {moderator.round} ---\n")
                f.write(f"Moderator: {mod_response}\n\n")
                print(f"\n-------- ROUND {moderator.round} ---------")
            
            # Los expertos responden
            dlb_response = dlb.generate(mod_response)
            f.write(f"DLB: {dlb_response}\n\n")
            
            pnm_response = pnm.generate(mod_response)
            f.write(f"PNM: {pnm_response}\n\n")
            
            if moderator.check_termination():
                # Cierre y síntesis técnica final
                f.write(f"\nFINAL DECISION\n")
                final_text = moderator.final_decision()
                f.write(f"Moderator (Final Decision): {final_text}\n")
                
                # Almacenamiento estructurado en JSON para análisis posterior
                if decision_dir:
                    import json
                    json_path = Path(decision_dir) / f"{slugify(topic)}.json"
                    data_json = {
                        "topic": topic,
                        "final_decision": final_text,
                        "rounds": moderator.round
                    }
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(data_json, jf, indent=4)
                break
    
    print(f"✅ ÉXITO: {outfile.name}")