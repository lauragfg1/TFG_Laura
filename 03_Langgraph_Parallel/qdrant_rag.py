import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder
import os

# --- Configuración UMA ---
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents_satcom_uma"
BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Plazas garantizadas por tipo de contenido en la recuperación inicial (antes
# de rerankear). El contenido "structured" (cifras/tablas) es solo ~11% del
# índice (1031 de 9352 puntos) y tiene un formato de texto muy distinto al
# narrativo, así que en una única búsqueda mixta de k=15 vecinos perdía
# sistemáticamente contra el narrativo y casi nunca aparecía en el contexto
# — que es probablemente la razón de fondo de que el juez no pueda puntuar
# alto respuestas que no citan cifras: el contexto rara vez las traía.
K_STRUCTURED = 5
K_NARRATIVE = 10

qclient = QdrantClient(url=QDRANT_URL)
bi_encoder = SentenceTransformer(BI_ENCODER_MODEL, device="cpu")
reranker = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")


def _query_by_type(query_vector, k, content_type):
    response = qclient.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(must=[FieldCondition(key="metadata.type", match=MatchValue(value=content_type))]),
        limit=k,
    )
    return response.points


def recuperar_contexto(query, topic="", k=15):
    try:
        enriched_query = f"Topic: {topic} | Question: {query} | metrics: Gain, SNR, Bandwidth, Loss, ms, Mbps"
        query_vector = bi_encoder.encode(enriched_query).tolist()

        # Dos búsquedas separadas (una por tipo) en vez de una mixta: garantiza
        # que el contenido estructurado tenga sitio en el contexto si existe
        # algún punto relevante, en vez de competir en desventaja por entrar
        # en un único top-k dominado por el narrativo.
        structured_points = _query_by_type(query_vector, K_STRUCTURED, "structured")
        narrative_points = _query_by_type(query_vector, K_NARRATIVE, "narrative")
        points = list(structured_points) + list(narrative_points)

        if not points: return "No context found."

        # RE-RANKING (sobre el conjunto combinado de ambos tipos)
        passages = [p.payload.get('content', '') for p in points]
        scores = reranker.predict([(query, p) for p in passages])

        scored_points = sorted(zip(scores, points), key=lambda x: x[0], reverse=True)

        structured = []
        narrative = []
        
        for score, p in scored_points:
            val_score = float(score)
            # Quitamos el filtro hardcoded de -2 porque los modelos Cross-Encoder arrojan logits (que suelen ser negativos en contextos muy específicos)
            
            payload = p.payload
            m = payload.get('metadata', {})
            content = payload.get('content', '')
            info = f"[Score: {val_score:.2f}][Source: {m.get('source','?')}, P.{m.get('page','?')}] {content}"

            if m.get('type') == "structured":
                structured.append(info)
            else:
                narrative.append(info)

        # FIX 2: Separación física de contexto (Grounding)
        contexto_final = "\n" + "="*40 + "\n"
        contexto_final += "[STRUCTURED DATA - HIGH PRIORITY]\n"
        contexto_final += ("\n".join(structured) if structured else "No structured data found.")
        contexto_final += "\n\n[NARRATIVE TEXT CONTEXT]\n"
        contexto_final += ("\n".join(narrative[:5]) if narrative else "No narrative data found.")
        contexto_final += "\n" + "="*40

        top_score = float(scored_points[0][0]) if scored_points else "NONE"
        logging.info(f"TOP SCORE: {top_score}")
        logging.info(f"📊 RAG: {len(structured)} tablas y {len(narrative[:5])} textos enviados al experto.")
        
        return contexto_final

    except Exception as e:
        logging.error(f"⚠️ Error RAG: {e}")
        return "Context error."