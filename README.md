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
qdrantDB/               Directorio de trabajo de la instancia local de Qdrant. El binario, el
                        dashboard web y el estado de la base de datos NO se versionan (ver
                        "Configurar Qdrant" más abajo); se generan/descargan en el primer arranque.
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
- Un servidor Qdrant 1.17.x local en `http://localhost:6333` — ver "Configurar Qdrant" abajo, no se distribuye en este repositorio.
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

## Configurar Qdrant

El binario de Qdrant, su dashboard web y el estado de la base de datos vectorial **no se
distribuyen en este repositorio** (son binarios/artefactos de terceros y estado regenerable,
no código del TFG — ver `.gitignore`). `qdrantDB/` es solo el directorio de trabajo donde se
instala y persiste la instancia local; el contenido se reconstruye siguiendo estos pasos.

Los experimentos de la memoria se hicieron con **Qdrant 1.17.0**. Se recomienda esa misma
versión para reproducibilidad exacta, aunque cualquier `1.17.x` debería ser compatible con
`qdrant-client>=1.16.1` del `requirements.txt`.

### Opción A — Docker (recomendada, multiplataforma)

```bash
docker run -d --name qdrant-tfg \
  -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrantDB/storage:/qdrant/storage" \
  qdrant/qdrant:v1.17.0
```

- `-p 6333:6333` expone la API HTTP/REST que usan todos los scripts (`http://localhost:6333`).
- `-v .../qdrantDB/storage:...` persiste la base de datos en el propio repo (ya excluido de git),
  para no perder el índice al reiniciar el contenedor.
- Para arrancarlo de nuevo en sesiones futuras: `docker start qdrant-tfg`.

### Opción B — Binario nativo (Windows, sin Docker)

1. Descargar el release `v1.17.0` para Windows desde las
   [releases oficiales de Qdrant](https://github.com/qdrant/qdrant/releases/tag/v1.17.0)
   (`qdrant-x86_64-pc-windows-msvc.zip`).
2. Extraer `qdrant.exe` dentro de `qdrantDB/` (la carpeta ya existe en el repo — ver
   `qdrantDB/README.md` — aunque ningún script depende de esa ruta directamente: Qdrant solo
   se referencia por su URL de red).
3. Arrancarlo desde la raíz del proyecto:
   ```powershell
   .\qdrantDB\qdrant.exe
   ```
   Por defecto persiste su estado en `./storage` relativo al directorio desde el que se lanza,
   por eso se recomienda ejecutarlo siempre desde `qdrantDB/` o pasar `--storage-dir`.

### Verificar que está levantado

```bash
curl http://localhost:6333/healthz
```

Debe responder `healthz check passed`. El dashboard web (si se usa Docker o el binario, ambos
lo sirven igual) está disponible en `http://localhost:6333/dashboard`.

## Preparar el RAG

Con Qdrant en marcha, indexar el corpus una vez:

> **Nota sobre `Jabega_documents/`:** esta carpeta no se distribuye en el repositorio
> público. Contiene el texto completo de 168 artículos científicos del dominio SATCOM
> descargados a través del acceso institucional de la Biblioteca de la Universidad de
> Málaga; al tratarse de material con derechos de autor de las editoriales originales, no
> puede redistribuirse fuera del ámbito académico para el que se obtuvo. Para reproducir el
> corpus, es necesario disponer de acceso propio a esos artículos (por ejemplo, a través de
> la biblioteca de tu propia institución) y colocarlos en `Jabega_documents/jabega/` en
> formato `.txt` antes de ejecutar `rag_indexer.py`.

```bash
venv/Scripts/python rag_indexer.py
```

Esto crea (o recrea desde cero, si ya existía) la colección `documents_satcom_uma` a partir de
`Jabega_documents/jabega/` — ver `rag_indexer.py`, que borra la colección si existe antes de
reindexar, así que es seguro volver a ejecutarlo. Al terminar debería reportar **9.352 puntos
vectoriales** (1.031 de contenido estructurado, ver Sección 4.1.2 de la memoria); si el número
difiere, revisa que `Jabega_documents/jabega/` esté completo.

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
- El binario de Qdrant, su dashboard y el estado de la base de datos vectorial no están versionados (ver "Configurar Qdrant"); todos los resultados de `data/`/`Resultados/` sí lo están, así que las figuras y tablas se pueden regenerar sin volver a levantar Qdrant ni a ejecutar los debates — solo hace falta Qdrant si se quiere repetir la generación desde cero.

## Autora

Laura Granda Fernández — TFG tutorizado por Sergio Gálvez Rojas, Departamento de Lenguajes y Ciencias de la Computación, Universidad de Málaga.
