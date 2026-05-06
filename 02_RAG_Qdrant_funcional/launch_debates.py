#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import time
import logging
from pathlib import Path

# --- Configuración del Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("debates_ejecucion.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout) # Mantiene la salida por consola
    ]
)

# Esto evita que la consola se llene de mensajes de conexión HTTP y descargas
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Obtiene la ruta de la carpeta donde está este script
script_dir = Path(__file__).parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Importamos la lógica desde el script base
try:
    from SGR_debate_Ollama import ejecutar_debate_completo, slugify
except ImportError:
    logging.error("No se encontró 'SGR_debate_Ollama.py' en el directorio actual.")
    sys.exit(1)

def iter_questions(txt_path: Path):
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            q = line.strip()
            if not q or q.startswith("#"): continue
            yield q

def main():
    if len(sys.argv) < 3:
        print("Uso: python launch_debates.py [FOLDER_TXT] [FOLDER_OUT] [OPTIONAL: FOLDER_JSON]")
        sys.exit(1)

    txt_dir = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    decision_dir = Path(sys.argv[3]).resolve() if len(sys.argv) >= 4 else None
    if decision_dir: decision_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(txt_dir.glob("*.txt"))
    
    logging.info("="*50)
    logging.info("INICIO DE SESIÓN DE DEBATES")
    logging.info(f"Directorio de temas: {txt_dir}")
    logging.info("="*50)

    total_q = 0
    exitos = 0
    fallos = 0
    start_session = time.time()
    for txt in txt_files:
        logging.info(f"📁 Procesando archivo: {txt.name}")
        
        for i, question in enumerate(iter_questions(txt), start=1):
            total_q += 1
            out_file = out_dir / f"{txt.stem}__q{i:02d}_{slugify(question)}.txt"

            logging.info(f"🚀 [{total_q}] Lanzando: {question[:60]}...")
            
            try:
                # Ejecución del debate
                ejecutar_debate_completo(
                    topic=question, 
                    outfile=out_file, 
                    decision_dir=decision_dir
                )
                logging.info(f"  ✅ ÉXITO: {out_file.name}")
                exitos += 1

            except Exception as e:
                fallos += 1
                logging.error(f"  ❌ FALLO en '{question[:40]}...': {str(e)}", exc_info=True)
    
    duracion = (time.time() - start_session) / 60
    logging.info("="*50)
    logging.info("RESUMEN DE LA SESIÓN")
    logging.info(f"Tiempo total: {duracion:.2f} minutos")
    logging.info(f"Debates totales: {total_q}")
    logging.info(f"Finalizados con éxito: {exitos}")
    logging.info(f"Errores encontrados: {fallos}")
    logging.info("="*50)

if __name__ == "__main__":
    main()