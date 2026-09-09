# Arquitecturas de orquestación multiagente para la resolución de problemas técnicos SATCOM

Trabajo de Fin de Grado — Grado en Ingeniería Informática (Mención en Sistemas de Información), Universidad de Málaga.

Compara tres arquitecturas de debate multiagente (individual, homogénea, heterogénea) sobre tres tamaños de modelo (2B / 8B / 70B) y tres frameworks de orquestación (**LangGraph**, **AutoGen**, **CrewAI**), aplicadas a la resolución de preguntas técnicas del dominio de comunicaciones por satélite (SATCOM). Las respuestas se generan con modelos ejecutados localmente vía **Ollama** y se enriquecen con **RAG** sobre un corpus de artículos científicos indexado en **Qdrant**. La calidad se evalúa automáticamente mediante un LLM-juez de 120B parámetros, y las comparaciones se contrastan estadísticamente con un test *t* de Student pareado.

Este repositorio contiene el código completo usado para generar los resultados de la memoria — no solo el código final, también los scripts de análisis y las figuras.

## Estructura del repositorio

```
01_Langgraph_Debate/    Implementación del sistema de debate en LangGraph (grafo de estados,
                        fan-out/fan-in DLB↔PNM, RAG, juez) + modo individual (baseline sin debate).
                        data/<2B|8B|70B>/<individual|homogeneo|heterogeneo>/repN/  — resultados
02_Autogen_Debate/      Implementación equivalente en AutoGen (GroupChat round-robin + síntesis
                        externa al canal para evitar cámara de eco).
                        Resultados/<2B|8B>/repN/
03_CrewAI_Debate/       Implementación equivalente en CrewAI (llamadas directas a Task.execute()
                        con historial pasado a mano, por incompatibilidad de Task.context en
                        crewai==0.1.24). Venv propio, ver más abajo.
                        Resultados/<2B|8B>/repN/
Dataset_Preguntas/      50 preguntas técnicas SATCOM (una por documento fuente), usadas por
                        igual en los tres frameworks.
Jabega_documents/       Corpus documental (artículos científicos SATCOM) que se indexa en Qdrant.
qdrantDB/               Binario e instancia local de Qdrant (base de datos vectorial).
figuras_tfg/            Figuras generadas para la memoria (PNG, 300 dpi).
rag_indexer.py          Indexa Jabega_documents/ en Qdrant (colección documents_satcom_uma).
generar_resumen_stats.py   Agrega las repeticiones de cada condición en un único CSV resumen.
generar_figuras_tfg.py     Genera las figuras de barras/comparativas de la Sección 5.
generar_heatmap_tamano_arquitectura.py  Heatmap 3×3 tamaño×arquitectura con significancia (Sección 5.2).
stats_comparativa.py      Comparativa rápida entre frameworks en la condición 8B heterogéneo.
judge_order_study.py      Estudio de sensibilidad del juez al orden del prompt (Sección 5.1.2).
judge_evaluator_robust.py Evaluador LLM-as-a-Judge con reintentos ante JSON inválido.
extraer_ejemplos_memoria.py  Extrae ejemplos reales (alta calidad / alucinación alta) citados en la memoria.
estimar_recarga_70b.py    Estima el tiempo de recarga de modelos en VRAM para la condición 70B heterogéneo.
```

## Requisitos

- Python 3.10 (probado también con 3.9 en el venv de CrewAI).
- [Ollama](https://ollama.com) corriendo en local (`http://localhost:11434`).
- Un servidor Qdrant local en `http://localhost:6333` (el binario está en `qdrantDB/qdrant.exe`, o usar Docker).
- GPU(s) con VRAM suficiente para los modelos que se quieran ejecutar. Los experimentos completos de la memoria se hicieron en un servidor con 3× GPU (72 GB VRAM total) — ver Sección 3.1 de la memoria.

### Modelos de Ollama necesarios

```bash
# Tamaño 2B (heterogéneo)
ollama pull gemma2:2b
ollama pull qwen2.5:1.5b
ollama pull deepseek-r1:1.5b

# Tamaño 8B (heterogéneo)
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b

# Tamaño 70B (heterogéneo)
ollama pull llama3.3:70b
ollama pull deepseek-r1:70b
ollama pull qwen2.5:72b

# Juez de evaluación
ollama pull gpt-oss:120b
```

En modo **homogéneo**, los tres roles (moderador/DLB/PNM) usan el mismo modelo (el indicado como moderador de cada tamaño).

### Entornos virtuales (dos, no compatibles entre sí)

`crewai==0.1.24` exige `langchain==0.0.354` exacto, incompatible con la versión moderna de LangGraph que usan los otros dos frameworks. Por eso hay **dos venvs separados**:

```bash
# Venv principal — para 01_Langgraph_Debate y 02_Autogen_Debate
python -m venv venv
venv/Scripts/pip install -r requirements.txt

# Venv propio de CrewAI — NO instalar esto en ../venv
python -m venv 03_CrewAI_Debate/venv
03_CrewAI_Debate/venv/Scripts/pip install -r 03_CrewAI_Debate/requirements.txt
```

## Preparar el RAG

Con Qdrant en marcha, indexar el corpus una vez:

```bash
venv/Scripts/python rag_indexer.py
```

Esto crea la colección `documents_satcom_uma` a partir de `Jabega_documents/jabega/`.

## Ejecutar los experimentos

Cada framework guarda sus resultados en su propia carpeta `data/`/`Resultados/`, organizados por `<tamaño>/<arquitectura>/repN/`.

**LangGraph — modo debate** (`--mode` homogeneo|heterogeneo):
```bash
venv/Scripts/python 01_Langgraph_Debate/launch_debate.py Dataset_Preguntas 01_Langgraph_Debate/data --size 8B --mode heterogeneo --rep 1
```

**LangGraph — modo individual** (baseline sin debate):
```bash
venv/Scripts/python 01_Langgraph_Debate/launch_single.py Dataset_Preguntas 01_Langgraph_Debate/data --size 8B --rep 1
```

**AutoGen:**
```bash
venv/Scripts/python 02_Autogen_Debate/autogen_debate.py --size 8B --rep 1
```

**CrewAI** (usando su propio venv):
```bash
03_CrewAI_Debate/venv/Scripts/python 03_CrewAI_Debate/crewai_debate.py --size 8B --rep 1
```

Todos aceptan `--start N` para reanudar desde una pregunta concreta si una ejecución se interrumpe.

## Evaluar con el juez

```bash
venv/Scripts/python 01_Langgraph_Debate/judge_evaluator.py 01_Langgraph_Debate/data/8B/heterogeneo/rep1
```

Genera `judge_evaluation_results.csv` dentro del directorio indicado. `judge_evaluator_robust.py` añade reintentos ante respuestas JSON inválidas del juez.

## Análisis y figuras

Una vez generados los datos y sus evaluaciones:

```bash
venv/Scripts/python generar_resumen_stats.py           # agrega las repeticiones -> resumen_metricas_todas_condiciones.csv
venv/Scripts/python generar_figuras_tfg.py              # figuras de barras (Sección 5)
venv/Scripts/python generar_heatmap_tamano_arquitectura.py  # heatmap con significancia (Sección 5.2)
venv/Scripts/python stats_comparativa.py                 # comparativa rápida entre frameworks
venv/Scripts/python extraer_ejemplos_memoria.py          # ejemplos reales citados en la memoria
```

Todos leen directamente de los `data/`/`Resultados/` generados en el paso anterior; ningún número de la memoria está escrito a mano.

## Notas de reproducibilidad

- El test de significancia usado en toda la memoria (test *t* de Student pareado por pregunta) se recalcula directamente desde los CSV de evaluación del juez en `generar_heatmap_tamano_arquitectura.py` — no depende de ningún cálculo intermedio no versionado.
- La comparativa completa de 9 condiciones (tamaño × arquitectura) se ejecutó únicamente sobre LangGraph; la comparativa entre frameworks se restringe a la condición 8B heterogénea en los tres (ver Sección 5.3 de la memoria y Líneas futuras).
- `03_CrewAI_Debate/Resultados/2B/` y `02_Autogen_Debate/Resultados/2B/` contienen una campaña adicional (2B en ambos frameworks) generada pero no incluida en el análisis de la memoria — punto de partida para la línea futura de extender la comparativa de frameworks a más tamaños.

## Autora

Laura Granda Fernández — TFG tutorizado por Sergio Gálvez Rojas, Departamento de Lenguajes y Ciencias de la Computación, Universidad de Málaga.
