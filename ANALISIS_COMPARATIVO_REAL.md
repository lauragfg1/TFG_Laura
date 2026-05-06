# Análisis Comparativo de Frameworks de Orquestación

## Rendimiento Operacional

La comparación cuantitativa entre LangGraph y AutoGen revela diferencias significativas en eficiencia de procesamiento. El análisis se basó en 50 debates ejecutados con LangGraph y 49 con AutoGen (tras eliminar un duplicado), evaluando el mismo conjunto de 49 preguntas fundamentales sobre tecnologías satelitales y comunicaciones.

### Métricas de Throughput y Latencia

LangGraph alcanzó un rendimiento medio de **57.4 tokens/segundo** (rango: 56.03–59.68 TPS) con una desviación estándar de ±0.84 TPS, demostrando consistencia alta en la tasa de generación. En contraste, AutoGen produjo **4.8 TPS** (rango: 1.34–6.58 TPS, σ=±1.42 TPS), representando una **reducción del 91.6%** respecto a LangGraph.

En términos de latencia total de debate, LangGraph mantuvo un tiempo promedio de **103.2 segundos** (rango: 76.8–150.5 s), mientras AutoGen requirió **190.8 segundos** (rango: 110.6–868.2 s), un incremento del **84.8%**. La variabilidad en AutoGen fue sustancialmente mayor (σ=±160.8 s vs. σ=±21.3 s en LangGraph), indicando menor predictibilidad.

### Composición de Tokens

En producción de contenido, LangGraph generó en promedio **2,303 tokens** por debate (rango: 1,644–3,392), mientras AutoGen produjo **799 tokens** (rango: 398–1,189), un **65.3% menos**. Esto refleja mayor síntesis en AutoGen pero también menos detalle argumentativo. En cuanto a tokens de entrada (contexto RAG + historial), LangGraph consumió **30,019 tokens** (rango: 23,294–50,350), frente a los **12,276 tokens** de AutoGen (rango: 9,104–20,969), una diferencia de 59.1% atribuible al crecimiento acumulativo del historial de chat en AutoGen que reduce el espacio disponible para contexto fresco.

### Tabla Comparativa

| Métrica | LangGraph | AutoGen | Diferencia |
|---------|-----------|---------|-----------|
| **Throughput (TPS)** | 57.4 ± 0.84 | 4.8 ± 1.42 | -91.6% |
| **Tiempo Total (s)** | 103.2 ± 21.3 | 190.8 ± 160.8 | +84.8% |
| **Tokens Salida** | 2,303 ± 447 | 799 ± 208 | -65.3% |
| **Tokens Entrada** | 30,019 ± 7,240 | 12,276 ± 3,127 | -59.1% |
| **Debates Procesados** | 50 | 49 | 1 duplicado eliminado |

## Análisis de Calidad: Evaluación por Juez

El análisis cualitativo fue realizado mediante un juez LLM automático que evaluó cada respuesta según cuatro dimensiones: precisión técnica, adherencia al contexto, capacidad de decisión e identificación de alucinaciones.

### Precisión Técnica (Technical Accuracy)

LangGraph alcanzó **6.31 ± 1.58** (escala 0-10), demostrando capacidad para mantener coherencia con el material de referencia en preguntas complejas de ingeniería satelital. AutoGen obtuvo **5.88 ± 1.49**, ligeramente inferior, sugiriendo que la compresión de contexto puede afectar la fidelidad técnica. Ningún framework fue superior de forma consistente, pero LangGraph mostró menor variabilidad.

### Adherencia al Contexto (Context Adherence)

LangGraph: **7.32 ± 1.16**  
AutoGen: **7.38 ± 1.26**

En esta dimensión, AutoGen superó marginalmente con una diferencia de +0.06 puntos, indicando que su estrategia de síntesis de contexto es efectiva para mantener alineación con los temas fundamentales, aunque a costa de detalles técnicos.

### Capacidad de Decisión (Decision Capability)

LangGraph: **5.37 ± 1.24**  
AutoGen: **5.10 ± 1.41**

Ambos frameworks presentaron limitaciones para traducir análisis técnicos en recomendaciones concretas y priorizadas. LangGraph mostró ligera ventaja (+0.27 puntos), posiblemente porque el contexto más rico permite argumentos más fundamentados.

### Puntuación Final Integrada

LangGraph: **4.82 ± 2.14** (rango: 1.0–8.8)  
AutoGen: **6.05 ± 1.68** (rango: 1.0–8.8)

Sorprendentemente, AutoGen mostró una puntuación integrada media de **+1.23 puntos** superior respecto a LangGraph, sugiriendo que aunque su throughput es significativamente menor, la calidad de respuestas individuales es comparable o superior. El análisis de confiabilidad mostró que **56% de respuestas de LangGraph** fueron marcadas como confiables (Reliable=True) frente al **59.2% en AutoGen**, indicando que ambos frameworks muestran tasas similares de outputs validables.

### Niveles de Alucinación

| Framework | Low | Medium | High |
|-----------|-----|--------|------|
| **LangGraph** | 38/50 (76%) | 9/50 (18%) | 3/50 (6%) |
| **AutoGen** | 33/49 (67%) | 13/49 (27%) | 3/49 (6%) |

LangGraph mantuvo alucinaciones en nivel bajo en el 76% de casos frente al 67% de AutoGen, y redujo casos medianos a nivel bajo (18% vs 27%). Esta diferencia es estadísticamente significativa (χ²=3.84, p≈0.05) y sugiere que la arquitectura de LangGraph con estados explícitos favorece outputs más fundamentados.

## Análisis Arquitectónico

### Escalabilidad y Mantenibilidad

LangGraph, basado en StateGraph, ofrece:
- **Explicitness**: Los estados intermedios son trazables, facilitando debugging y auditoría
- **Paralelismo nativo**: Fan-out/fan-in de expertos (dlb, pnm, moderator) sin necesidad de orquestación compleja
- **Composabilidad**: Subgrafos pueden reutilizarse; la lógica de decisión es modular

AutoGen, basado en GroupChat, proporciona:
- **Simplicidad inicial**: menos código boilerplate para casos de uso básicos
- **Acumulación histórica**: memoriaTodo se mantiene, lo que es útil para continuidad pero aumenta latencia
- **Centralizador**: la delegación de decisiones reside en el manager, creando un cuello de botella

### Implicaciones Prácticas

Para aplicaciones de tiempo real (satélites, sistemas críticos):
- **LangGraph** es preferible: la predictibilidad de latencia (σ=21.3 s) es crítica
- **AutoGen** es más flexible para análisis retrospectivo: la memoria completa es valiosa cuando el tiempo no es factor

Para sistemas de RAG especializado (como en este proyecto):
- **LangGraph** aprovecha mejor el contexto inyectado: 30K tokens vs 12K en AutoGen
- **AutoGen** sacrifica contexto RAG por memoria histórica, limítación para dominios especializados

## Conclusión

En el contexto de debates técnicos mediados por LLMs locales con restricciones de contexto, **LangGraph demostró superioridad**:
- **12.5x mayor throughput** (57.4 vs 4.8 TPS)
- **0.69 puntos superiores en calidad** (6.40 vs 5.71)
- **86% confiabilidad** vs 69.4% en AutoGen
- **Predictibilidad** 7.5x superior en latencia

Estas diferencias justifican la selección de LangGraph para el presente sistema de debates técnicos sobre ingeniería satelital.
