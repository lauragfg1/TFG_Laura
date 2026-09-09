#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import json
import csv
import re
import unicodedata
from pathlib import Path
import argparse

# Importamos las herramientas de LangChain y tus módulos
from langchain_core.messages import HumanMessage
from ollama_client import call_ollama
from qdrant_rag import recuperar_contexto
from dataset_questions import iter_questions, source_files

# --- Configuración del Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("single_ejecucion.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Configuración de Modelos Individuales ---
# Usamos el modelo principal (el que hace de Moderador en los debates)
SINGLE_MODELS = {
    "2B": "gemma2:2b",
    "8B": "llama3.1:8b",
    "70B": "llama3.3:70b"
}

# --- Utilidades ---
def slugify(value):
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '_', value)[:60]

# --- Función de Guardado ---
def save_single_results(q_id, question, model_id, result, q_folder, csv_path, context, rep=1):
    try:
        model_json = json.loads(result["content"])
        final_response = model_json.get("response", result["content"])
        final_context = model_json.get("reference_context", context)
    except Exception:
        final_response = result["content"]
        final_context = context

    # 1. Guardar TXT
    txt_path = q_folder / f"response_single_q{q_id:03d}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"ID: {q_id}\nTOPIC: {question}\nMODEL: {model_id}\n")
        f.write("="*50 + "\n")
        f.write(final_response + "\n")

    # 2. Guardar JSON exacto a tus requisitos
    json_data = {
        "topic": question,
        "response": final_response,
        "reference_context": final_context
    }
    json_path = q_folder / f"decision_{q_id:03d}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    # 3. Procesamiento de las métricas (CSV)
    m = result.get("metrics")

    # Detalle de la llamada (tiempos, num_predict/temperatura reales usados) para
    # diagnóstico posterior, igual que en el modo debate.
    metrics_path = q_folder / f"metrics_{q_id:03d}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump([m] if isinstance(m, dict) else [], f, indent=2)

    if m and isinstance(m, dict):
        total_s = m.get("tiempo_total_s", 0)
        load_s = m.get("tiempo_carga_s", 0)
        prompt_s = m.get("tiempo_prompt_eval_s", 0)
        gen_s = m.get("tiempo_generacion_s", 0)
        overhead_s = m.get("tiempo_overhead_s", 0)
        t_in = m.get("tokens_prompt", 0)
        t_out = m.get("tokens_generacion", 0)
        avg_tps = m.get("tps", 0)
    else:
        total_s = load_s = prompt_s = gen_s = overhead_s = t_in = t_out = avg_tps = 0

    # 4. Escribir en el CSV
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "ID", "Rep", "Size", "Mode", "Model", "Total_S", "Load_S", "Prompt_S",
                "Gen_S", "Overhead_S", "Tokens_In", "Tokens_Out", "TPS", "Topic"
            ])

        writer.writerow([
            q_id,
            rep,
            os.environ.get("SIZE", "unknown"),
            "single",
            model_id,
            f"{total_s:.3f}",
            f"{load_s:.3f}",
            f"{prompt_s:.3f}",
            f"{gen_s:.3f}",
            f"{overhead_s:.3f}",
            t_in,
            t_out,
            f"{avg_tps:.2f}",
            question[:100]
        ])

def main():
    parser = argparse.ArgumentParser(description="Lanzador de respuestas individuales (Baseline) para SatCom")
    parser.add_argument("input", help="Carpeta con archivos .txt (Dataset)")
    parser.add_argument("output", help="Carpeta base /data donde se guardarán los resultados")
    parser.add_argument("--size", required=True, choices=["2B", "8B", "70B"], help="Tamaño del modelo a evaluar")
    parser.add_argument("--limit", type=int, default=None, help="Número máximo de preguntas a procesar")
    parser.add_argument("--start", type=int, default=1, help="Número de pregunta por la que empezar")
    parser.add_argument("--rep", type=int, default=1, help="Número de repetición del experimento (crea data/.../repN/)")

    args = parser.parse_args()

    # Sincronizar variables de entorno para que ollama_client ajuste hilos y timeout
    os.environ["SIZE"] = args.size
    os.environ["MODE"] = "single"  # Fuerza modo single

    model_id = SINGLE_MODELS[args.size]

    # Creación de carpetas (Ej: data/70B/individual/rep1/), una por repetición
    base_dir = Path(args.output).resolve()
    out_dir = base_dir / args.size / "individual" / f"rep{args.rep}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "experiment_metrics.csv"
    input_path = Path(args.input)
    txt_files = source_files(input_path)

    total_q = 0

    for txt in txt_files:
        logging.info(f"📁 Procesando archivo: {txt.name}")
        for i, question in enumerate(iter_questions(txt), start=1):
            total_q += 1
            
            if total_q < args.start:
                continue

            if args.limit and total_q > args.limit:
                logging.info(f"🛑 Límite de {args.limit} preguntas alcanzado. Deteniendo...")
                return

            # Crear subcarpeta individual (Ej: data/70B/individual/q007)
            q_folder = out_dir / f"q{total_q:03d}"
            q_folder.mkdir(parents=True, exist_ok=True)

            logging.info(f"👉 [{total_q}] Generando respuesta individual con {model_id}...")
            
            try:
                # 1. Recuperar contexto RAG
                context = recuperar_contexto(question)
                
                # 2. Generar el Prompt
                prompt = f"""You are a highly skilled Senior SATCOM Engineer. Answer the following technical question based on the provided context.

                Context: {context}

                Question: {question}

                Requirements:
                - Be technically precise.
                - Use metrics and specific standards ONLY if they are explicitly present in the Context.
                - DO NOT invent or fabricate any metric, number, or system fact that is not stated in the Context. If the Context lacks data, explicitly state so, and rely strictly on universal physics principles to deduce the answer.
                - Provide a direct answer without conversational filler.
                - Return ONLY VALID JSON. Do not write anything outside the JSON structure.
                
                OUTPUT FORMAT:
                {{
                    "response": "Your technical answer here. Max 100 words.",
                    "reference_context": "Write a 60-word summary of the factual ground-truth extracted from the reference context provided above."
                }}
                """
                
                # 3. Llamar a Ollama
                result = call_ollama(model_id, [HumanMessage(content=prompt)], force_json=True)
                
                if result and result.get("content"):
                    # 4. Guardar resultados
                    save_single_results(total_q, question, model_id, result, q_folder, csv_path, context, rep=args.rep)
                    logging.info("   ✅ Respuesta guardada con éxito.")
                else:
                    logging.error(f"   ❌ Error: No se obtuvo respuesta del modelo para q{total_q}")

            except Exception as e:
                logging.error(f"   ❌ Error en la ejecución de q{total_q}: {str(e)}")
            
            time.sleep(1)

if __name__ == "__main__":
    main()