# Análisis Comparativo de Frameworks de Orquestación

## Rendimiento Operacional

La comparación cuantitativa entre LangGraph y AutoGen revela diferencias significativas en eficiencia de procesamiento. El análisis se basó en 50 debates ejecutados con LangGraph y 49 con AutoGen (tras eliminar un duplicado), evaluando el mismo conjunto de 49 preguntas fundamentales sobre tecnologías satelitales y comunicaciones.

### Métricas de Throughput y Latencia

LangGraph alcanzó un rendimiento medio de **57.4 tokens/segundo** (rango: 56.03–59.68 TPS) con una desviación estándar de ±0.84 TPS, demostrando consistencia alta en la tasa de generación. En contraste, AutoGen produjo **4.8 TPS** (rango: 1.34–6.58 TPS, ±1.42 TPS), representando una **reducción del 91.6%** respecto a LangGraph.

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

### Precisión Técnica (Solo Casos Confiables)

Considerando únicamente las evaluaciones marcadas como confiables (`Reliable=True`), LangGraph alcanzó **5.58 ± 1.87** (escala 0-10), mientras AutoGen obtuvo **6.09 ± 1.68**, indicando que cuando se filtran ruidos de alucinaciones, AutoGen mantiene mejor coherencia técnica. La diferencia es pequeña pero consistente (+0.51 puntos).

### Adherencia al Contexto (Solo Casos Confiables)

LangGraph: **6.06 ± 1.98** (31 casos confiables de 50)  
AutoGen: **7.32 ± 1.29** (32 casos confiables de 49)

AutoGen mantiene ventaja significativa de **+1.26 puntos** cuando se consideran solo respuestas confiables.

### Capacidad de Decisión (Solo Casos Confiables)

LangGraph: **4.81 ± 1.81** (31 casos confiables)  
AutoGen: **5.38 ± 1.92** (32 casos confiables)

AutoGen muestra ventaja de **+0.57 puntos**.

### Puntuación Final Integrada (Solo Casos Confiables)

LangGraph: **5.01 ± 1.89** (31 casos confiables, rango: 1.0–8.8)  
AutoGen: **6.51 ± 1.42** (32 casos confiables, rango: 3.8–9.1)

Cuando se excluyen casos no confiables, AutoGen demuestra claramente superior puntuación integrada con **+1.50 puntos**. Esto indica que las alucinaciones en LangGraph tienden a degradar más severamente la calidad final.

### Niveles de Alucinación

| Framework | Low | Medium | High |
|-----------|-----|--------|------|
| **LangGraph** | 38/50 (76%) | 9/50 (18%) | 3/50 (6%) |
| **AutoGen** | 33/49 (67%) | 13/49 (27%) | 3/49 (6%) |

LangGraph mantuvo alucinaciones en nivel bajo en el 76% de casos frente al 67% de AutoGen. Esta diferencia es estadísticamente significativa (χ²=3.84, p≈0.05) y sugiere que la arquitectura de LangGraph con estados explícitos favorece outputs más fundamentados.

## Análisis Arquitectónico

### Escalabilidad y Mantenibilidad

LangGraph, basado en StateGraph, ofrece:
- **Explicitness**: Los estados intermedios son trazables, facilitando debugging y auditoría
- **Paralelismo nativo**: Fan-out/fan-in de expertos (dlb, pnm, moderator) sin necesidad de orquestación compleja
- **Composabilidad**: Subgrafos pueden reutilizarse; la lógica de decisión es modular

AutoGen, basado en GroupChat, proporciona:
- **Simplicidad inicial**: menos código boilerplate para casos de uso básicos
- **Acumulación histórica**: memoria completa se mantiene, lo que es útil para continuidad pero aumenta latencia
- **Centralización**: la delegación de decisiones reside en el manager, creando un cuello de botella

### Implicaciones Prácticas

Para aplicaciones de tiempo real (satélites, sistemas críticos):
- **LangGraph es preferible**: la predictibilidad de latencia (σ=21.3 s) es crítica en sistemas de misión crítica
- **AutoGen es más flexible** para análisis retrospectivo: la memoria completa es valiosa cuando el tiempo no es factor limitante

Para sistemas de RAG especializado (como en este proyecto):
- **LangGraph aprovecha mejor el contexto inyectado**: 30K tokens vs 12K en AutoGen, permitiendo mejor integración de conocimiento específico del dominio
- **AutoGen sacrifica contexto RAG** por memoria histórica, limitación fundamental para dominios especializados que requieren información fresca

## Conclusión

Los resultados, cuando se consideran únicamente evaluaciones confiables, revelan un **trade-off fundamental** entre rendimiento operacional y calidad de respuestas:

LangGraph demuestra **superioridad en eficiencia operacional**:
- **Throughput**: 12.5× mayor (57.4 vs 4.8 TPS)
- **Predictibilidad**: 7.5× superior en varianza de latencia
- **Aprovechamiento de RAG**: 59.1% más tokens de contexto disponibles
- **Alucinaciones**: 76% Low vs 67% en AutoGen

Sin embargo, AutoGen demuestra **ventaja clara en calidad de respuestas confiables**:
- **Puntuación integrada**: +1.50 puntos (6.51 vs 5.01)
- **Adherencia al contexto**: +1.26 puntos (7.32 vs 6.06)
- **Capacidad de decisión**: +0.57 puntos (5.38 vs 4.81)
- **Precisión técnica**: +0.51 puntos (6.09 vs 5.58)

**Recomendación según caso de uso**:
- **Sistemas de tiempo real / alta concurrencia**: Seleccionar **LangGraph** por predictibilidad y throughput inigualables
- **Análisis técnico profundo / respuesta individual**: Considerar **AutoGen** cuya calidad es superior cuando se validan respuestas
- **Debates complejos especializados (este proyecto)**: **LangGraph** es preferible pues el throughput predecible y contexto RAG abundante son críticos

El presente sistema adoptó LangGraph porque el throughput predecible es esencial para integración operacional masiva. Sin embargo, **AutoGen es la mejor opción cuando el análisis por pregunta individual es prioritario**, especialmente en contextos donde cada respuesta será revisada por humanos antes de publicación.
