import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

# Esto asegura la conectividad directa con el servicio de Qdrant en localhost.
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

# --- Configuración 
QDRANT_URL = "http://localhost:6333" 
COLLECTION_NAME = "documents_satcom_uma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
DATA_PATH = "./Jabega_documents/jabega"

def rag_indexer():
    # 1. INICIALIZACIÓN
    if not os.path.isdir(DATA_PATH) or not any(f.endswith('.txt') for f in os.listdir(DATA_PATH)):
        raise FileNotFoundError(
            f"No se encuentra el corpus en '{DATA_PATH}'. Esta carpeta no se distribuye "
            "en el repositorio por derechos de autor editorial: ver la nota en README.md, "
            "seccion 'Preparar el RAG', para saber como reproducirla."
        )

    client = QdrantClient(url=QDRANT_URL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # 2. RECREAR COLECCIÓN
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # 3. PROCESAMIENTO
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    
    current_id = 0 
    print(f"--- Iniciando Ingesta Híbrida (Total: {len(files)} documentos) ---\n")

    for txt_file in files:
        # --- RESETEO DE VARIABLES POR ARCHIVO ---
        base_name = txt_file.replace('.txt', '')
        txt_path = os.path.join(DATA_PATH, txt_file)
        json_path = os.path.join(DATA_PATH, base_name + '_tables_tables.json')
        
        document_points = []
        txt_chunks_count = 0
        json_rows_count = 0

        # --- FASE A: PROCESAMIENTO DE TEXTO (.txt) ---
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = text_splitter.split_text(content)
        txt_chunks_count = len(chunks)
        
        for i, chunk in enumerate(chunks):
            vector = model.encode(chunk).tolist()
            document_points.append(PointStruct(
                id=current_id,
                vector=vector,
                payload={
                    "content": chunk,
                    "metadata": {"source": txt_file, "type": "narrative", "index": i}
                }
            ))
            current_id += 1 

        # --- FASE B: PROCESAMIENTO DE TABLAS (.json) 
        if os.path.exists(json_path) and os.path.getsize(json_path) > 5:
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data_json = json.load(f)
                    if data_json:
                        # 1. DEFINIMOS LOS TÉRMINOS IMPORTANTES AQUÍ (Para evitar el error)
                        important_terms = ["Gain", "Bandwidth", "SNR", "Eb/N0", "Loss", "Efficiency", "Frequency"]
                        
                        for table in data_json:
                            page = table.get("page", 0)
                            rows = table.get("data", [])
                            if len(rows) < 2: continue

                            # 2. DETECCIÓN DE HEADERS
                            def is_header_row(row_values):
                                text_cells, numeric_cells = 0, 0
                                for v in row_values:
                                    clean_v = str(v).strip()
                                    if re.search(r'\d', clean_v): numeric_cells += 1
                                    if re.search(r'[a-zA-Z]', clean_v): text_cells += 1
                                return text_cells >= numeric_cells

                            def clean_val(v):
                                if v is None: return ""
                                v = str(v).replace('\n', ' ').strip()
                                v = re.sub(r'\(cid:\d+\)', '', v)
                                return v

                            first_row_vals = [clean_val(v) for v in rows[0].values()]
                            is_header = is_header_row(first_row_vals)
                            
                            headers = {}
                            if is_header:
                                headers = {k: clean_val(v) for k, v in rows[0].items()}
                                data_rows = rows[1:]
                            else:
                                headers = {k: f"Param_{k}" for k in rows[0].keys()}
                                data_rows = rows

                            # 3. PROCESADO DE FILAS CON BOOSTING SEMÁNTICO
                            for row in data_rows:
                                pairs = []
                                for k, v in row.items():
                                    val = clean_val(v)
                                    if not val: continue
                                    
                                    header_name = headers.get(k, k)
                                    # Aplicamos el prefijo sugerido por Chati
                                    is_important = any(term.lower() in header_name.lower() for term in important_terms)
                                    prefix = "CRITICAL_METRIC: " if is_important else ""
                                    
                                    pairs.append(f"{prefix}{header_name}: {val}")

                                if pairs:
                                    row_str = " | ".join(pairs)
                                    technical_content = f"""
                                        [STRUCTURED_TABLE_DATA]
                                        Domain: SatCom / RF Engineering
                                        Source: {base_name}
                                        Page: {page}
                                        Data: {row_str}
                                        """.strip()

                                    vector_json = model.encode(technical_content).tolist()
                                    document_points.append(PointStruct(
                                        id=current_id, 
                                        vector=vector_json,
                                        payload={
                                            "content": technical_content,
                                            "metadata": {"source": os.path.basename(json_path), "type": "structured", "page": page}
                                        }
                                    ))
                                    current_id += 1
                                    json_rows_count += 1
                except Exception as e:
                    logging.error(f"⚠️ Error en {base_name}: {e}")

        # --- FASE C: CARGA ---
        if document_points:
            client.upsert(collection_name=COLLECTION_NAME, points=document_points)
            print(f"📄 Documento: {base_name}")
            print(f"   ├─ [Texto: {txt_chunks_count} chunks] | [Tablas: {json_rows_count} filas]")
            print(f"   └─ Total documento: {len(document_points)} puntos indexados.")
            print("-" * 20)

    print("\n" + "="*50)
    print(f"FINALIZADO: Total de {current_id} puntos indexados.")
    print("="*50)

if __name__ == "__main__":
    rag_indexer()