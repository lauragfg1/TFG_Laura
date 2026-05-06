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
from graph import app 
import argparse

# --- Configuración del Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("debates_ejecucion.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.getLogger("httpx").setLevel(logging.WARNING)

# --- Utilidades ---
def slugify(value):
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '_', value)[:60]

def iter_questions(txt_path: Path):
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            q = line.strip()
            if not q or q.startswith("#"): continue
            yield q

def extract_judge_json(text):
    try:
        # 1. Intentamos buscar el bloque tal cual
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        
        # 2. Reparación robusta (fallback)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
            
    except Exception as e:
        print(f"⚠️ Error intentando reparar el JSON: {e}")
    return None

# --- Función de Guardado (✅ Limpia sin decision_dir) ---
def save_results(debate_id, question, state, out_file, csv_path):
    # 1. Guardar Log TXT completo 
    with out_file.open("w", encoding="utf-8") as f:
        f.write(f"DEBATE ID: {debate_id}\nTOPIC: {question}\n" + "="*50 + "\n")
        for m in state["messages"]:
            f.write(f"\n{m.content}\n" + "-"*30)

    # 2. Extraer JSON del Moderador (Se guarda junto al TXT en la misma carpeta)
    judge_msg = state["messages"][-1].content
    judge_data = extract_judge_json(judge_msg)
    
    if judge_data:
        json_path = out_file.parent / f"decision_{debate_id:03d}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(judge_data, f, indent=4)

    # 3. Procesamiento de las métricas
    metrics_list = state.get("metrics", [])
    valid_metrics = [m for m in metrics_list if m is not None and isinstance(m, dict)]
    
    if valid_metrics:
        total_s = sum(m.get("tiempo_total_s", 0) for m in valid_metrics)
        load_s = sum(m.get("tiempo_carga_s", 0) for m in valid_metrics)
        prompt_s = sum(m.get("tiempo_prompt_eval_s", 0) for m in valid_metrics)
        gen_s = sum(m.get("tiempo_generacion_s", 0) for m in valid_metrics)
        
        t_in = sum(m.get("tokens_prompt", 0) for m in valid_metrics)
        t_out = sum(m.get("tokens_generacion", 0) for m in valid_metrics)
        
        avg_tps = sum(m.get("tps", 0) for m in valid_metrics) / len(valid_metrics)
    else:
        total_s = load_s = prompt_s = gen_s = t_in = t_out = avg_tps = 0
    
    # 4. Actualizar CSV con las nuevas columnas
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow([
                "ID", "Size", "Mode", "Winner", "Total_S", "Load_S", "Prompt_S", 
                "Gen_S", "Tokens_In", "Tokens_Out", "Avg_TPS", "Topic"
            ])
        
        writer.writerow([
            debate_id, 
            os.environ.get("SIZE", "unknown"),
            os.environ.get("MODE", "unknown"),
            judge_data.get("winner", "Unknown") if judge_data else "Error",
            f"{total_s:.3f}", 
            f"{load_s:.3f}", 
            f"{prompt_s:.3f}", 
            f"{gen_s:.3f}",
            t_in,
            t_out,
            f"{avg_tps:.2f}", 
            question[:100]
        ])

def main():
    # ✅ Se eliminó el argumento fantasma "decision" del parser
    parser = argparse.ArgumentParser(description="Lanzador de debates LangGraph para SatCom")
    parser.add_argument("input", help="Carpeta con archivos .txt o una pregunta directa entre comillas")
    parser.add_argument("output", help="Carpeta donde se guardarán los resultados (.txt y .csv)")
    parser.add_argument("--size", required=True, choices=["2B", "8B", "70B"], help="Tamaño de los modelos")
    parser.add_argument("--mode", required=True, choices=["heterogeneo", "homogeneo"], help="Modo de debate")
    parser.add_argument("--limit", type=int, default=None, help="Número máximo de preguntas a procesar en modo carpeta")
    parser.add_argument("--start", type=int, default=1, help="Número de debate por el que empezar") 

    args = parser.parse_args()
    
    os.environ["SIZE"] = args.size
    os.environ["MODE"] = args.mode

    input_arg = args.input
    
    # CREACIÓN DINÁMICA DE CARPETAS
    base_dir = Path(args.output).resolve()
    out_dir = base_dir / args.size / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir / "experiment_metrics.csv"

    is_directory = False
    try:
        if os.path.isdir(input_arg):
            is_directory = True
    except:
        is_directory = False

    # MODO PREGUNTA ÚNICA 
    if not is_directory:
        question = input_arg
        logging.info(f"🚀 Modo pregunta única detectado")
        out_file = out_dir / f"test_{slugify(question)}.txt"
        
        try:
            inputs = {"topic": question, "round": 0, "max_rounds": 5, "messages": [], "metrics": []}
            print(f"\n{'='*30}\nINICIANDO DEBATE ÚNICO: {question[:50]}...\n{'='*30}")
            final_state = app.invoke(inputs)
            if final_state and len(final_state["messages"]) > 0:
                save_results(1, question, final_state, out_file, csv_path) # ✅ Llamada limpia
                logging.info(f"✅ Debate finalizado con éxito.")
        except Exception as e:
            logging.error(f"❌ Error en la ejecución: {str(e)}")

    # MODO DATASET 
    else:
        input_path = Path(input_arg)
        txt_files = sorted(input_path.glob("*.txt"))
        total_q = 0
        
        for txt in txt_files:
            logging.info(f"📁 Procesando archivo: {txt.name}")
            for i, question in enumerate(iter_questions(txt), start=1):

                total_q += 1
                if total_q < args.start:
                    continue

                if args.limit and total_q >= args.limit:
                    logging.info(f"🛑 Límite de {args.limit} preguntas alcanzado. Deteniendo...")
                    return 

                # CARPETA ESPECÍFICA DE LA PREGUNTA
                q_folder = out_dir / f"q{total_q:03d}"
                q_folder.mkdir(parents=True, exist_ok=True)
                #out_file = q_folder / f"{txt.stem}__q{i:02d}_{slugify(question)}.txt"
                out_file = q_folder / f"debate_q{total_q:03d}.txt"
                
                logging.info(f"👉 [{total_q}] Iniciando debate...")
                try:
                    inputs = {"topic": question, "round": 0, "max_rounds": 5, "messages": [], "metrics": []}
                    final_state = app.invoke(inputs)
                    if final_state and len(final_state["messages"]) > 0:
                        save_results(total_q, question, final_state, out_file, csv_path) # ✅ Llamada limpia
                        logging.info(f"   ✅ Finalizado.")
                except Exception as e:
                    logging.error(f"   ❌ Error en debate #{total_q}: {str(e)}")
                
                time.sleep(1) # Pausa técnica para las GPUs

if __name__ == "__main__":
    main()