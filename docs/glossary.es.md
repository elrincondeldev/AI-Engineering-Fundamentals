🌐 [English](glossary.md) · Español

# Glossary · Glosario

Todos los términos clave del curso, con su equivalente en español. Están ordenados alfabéticamente
por el nombre en inglés, para que los encuentres igual en las dos versiones. Cuando en español se
suele mantener la palabra inglesa, la columna "Español" lo indica en cursiva.

## A

| Término (inglés) | Español | Significado |
|---|---|---|
| A2A (Agent2Agent protocol) | protocolo A2A | Protocolo abierto para que agentes creados por equipos distintos se descubran e intercambien tareas. |
| Abstention | abstención | Responder "no lo sé" (o pasar el caso a una persona) en lugar de adivinar cuando faltan pruebas. |
| Adapter | adaptador | Pequeño módulo entrenable que se inserta en un modelo congelado para adaptarlo a una tarea. |
| Agent | agente | Sistema en el que un modelo decide, en bucle, qué pasos y herramientas usar para lograr un objetivo. |
| Agentic system | sistema agéntico | Cualquier sistema que usa un LLM con cierta autonomía: desde *workflows* fijos hasta agentes autónomos. |
| Approximate nearest neighbour (ANN) | vecino más cercano aproximado | Búsqueda que sacrifica un poco de *recall* a cambio de una búsqueda vectorial mucho más rápida (IVF, HNSW, PQ). |
| Attention | atención | La operación del Transformer que permite a cada *token* ponderar información de otros *tokens*. |

## B

| Término (inglés) | Español | Significado |
|---|---|---|
| Base model | modelo base | Un modelo solo preentrenado: continúa texto, pero no sigue instrucciones de forma fiable. |
| Beam search | búsqueda en haz (*beam search*) | Decodificación que conserva en cada paso las `k` secuencias parciales más probables. |
| Best-of-N | mejor de N (*best-of-N*) | Generar N respuestas y quedarse con la que mejor puntúa un verificador o un modelo de recompensa. |
| BM25 | BM25 | Función clásica de ranking de texto completo basada en la frecuencia de términos, su rareza y la longitud del documento. |
| Byte-pair encoding (BPE) | codificación por pares de bytes (BPE) | Tokenizador que fusiona repetidamente el par de símbolos más frecuente en un *token* nuevo. |

## C

| Término (inglés) | Español | Significado |
|---|---|---|
| Chain of thought (CoT) | cadena de pensamiento | Pasos de razonamiento intermedios escritos antes de la respuesta final. |
| Chat template | plantilla de chat | El formato de texto exacto que convierte una lista de mensajes en la entrada del modelo. |
| Chunking | fragmentación (*chunking*) | Dividir documentos en trozos recuperables (*chunks*). |
| Citation | cita | Referencia desde una respuesta al pasaje de la fuente que la respalda. |
| Compaction | compactación | Reducir el contexto (resúmenes, poda, deduplicación) para que quepa en un presupuesto de *tokens*. |
| Concept drift | deriva de concepto | Un cambio con el tiempo en la relación entre las entradas y las etiquetas. |
| Context engineering | ingeniería de contexto | Decidir qué entra en el contexto del modelo, en qué orden y cuánto. |
| Context window | ventana de contexto | El número máximo de *tokens* que un modelo puede leer en una llamada. |
| Contamination | contaminación | Datos de evaluación que se colaron en los datos de entrenamiento e inflan las puntuaciones. |
| Cross-entropy | entropía cruzada | La pérdida con la que se entrenan los modelos de lenguaje: cuánto le sorprende al modelo el siguiente *token* real. |

## D

| Término (inglés) | Español | Significado |
|---|---|---|
| Data leakage | fuga de datos (*data leakage*) | Información en las entradas que no estaría disponible al predecir; a menudo, la propia etiqueta. |
| Decoder-only | solo decodificador (*decoder-only*) | La variante del Transformer que usan casi todos los LLM: predice cada *token* a partir de los anteriores. |
| Decoding | decodificación | Convertir las probabilidades del siguiente *token* en texto (voraz, en haz, muestreo). |
| Deduplication | deduplicación | Eliminar documentos duplicados exactos y casi duplicados. |
| Deep research | investigación profunda (*deep research*) | Un agente que planifica, busca, lee y escribe un informe con citas a partir de muchas fuentes. |
| Direct Preference Optimization (DPO) | optimización directa de preferencias (DPO) | Entrenar con pares de preferencias sin un modelo de recompensa ni un bucle de aprendizaje por refuerzo aparte. |
| Distillation | destilación | Entrenar un modelo más pequeño para imitar las salidas de uno más grande. |

## E

| Término (inglés) | Español | Significado |
|---|---|---|
| Embedding | *embedding* (vector de representación) | Un vector que representa un texto de modo que los significados parecidos quedan cerca. |
| Evaluation harness | arnés de evaluación | Código que ejecuta un sistema sobre un dataset fijo y reporta métricas de forma reproducible. |
| Exact match (EM) | coincidencia exacta | Métrica: 1 si la respuesta normalizada es igual a la de referencia; si no, 0. |

## F

| Término (inglés) | Español | Significado |
|---|---|---|
| FAISS | FAISS | Biblioteca de código abierto para búsqueda eficiente por similitud de vectores. |
| Faithfulness | fidelidad | Si cada afirmación de una respuesta está respaldada por el contexto proporcionado. |
| Few-shot prompting | *prompting* con pocos ejemplos (*few-shot*) | Incluir en el *prompt* unos pocos ejemplos resueltos. |
| Fine-tuning | ajuste fino (*fine-tuning*) | Continuar el entrenamiento de un modelo preentrenado con datos de la tarea. |

## G

| Término (inglés) | Español | Significado |
|---|---|---|
| GGUF | GGUF | Formato de archivo para pesos de modelos (a menudo cuantizados) que usan llama.cpp y Ollama. |
| Greedy decoding | decodificación voraz (*greedy*) | Elegir siempre el siguiente *token* más probable. |
| Grounding | fundamentación (*grounding*) | Basar una respuesta en las fuentes proporcionadas y no en la memoria del modelo. |
| Guardrail | barrera de protección (*guardrail*) | Una comprobación antes o después del modelo que bloquea o corrige entradas o salidas inseguras. |

## H

| Término (inglés) | Español | Significado |
|---|---|---|
| Hallucination | alucinación | Una salida fluida que no está respaldada por hechos ni fuentes. |
| HNSW | HNSW | *Hierarchical Navigable Small World*: un índice ANN basado en grafos. |
| Hybrid search | búsqueda híbrida | Combinar resultados de búsqueda por palabras clave y vectorial, por ejemplo con RRF. |

## I

| Término (inglés) | Español | Significado |
|---|---|---|
| Inference | inferencia | Ejecutar un modelo entrenado para producir salidas. |
| Inference-time scaling | escalado en tiempo de inferencia | Gastar más cómputo al responder (más muestras, búsqueda, razonamiento más largo) para mejorar la exactitud. |
| Instruction tuning | ajuste por instrucciones | Ajuste fino supervisado con pares de instrucción y respuesta. |
| IVF (inverted file index) | índice de archivo invertido (IVF) | Un índice ANN que agrupa los vectores y solo busca en los grupos más cercanos. |

## J

| Término (inglés) | Español | Significado |
|---|---|---|
| JSON Schema | JSON Schema | Un estándar para describir la forma de unos datos JSON; se usa para definir los parámetros de las herramientas. |

## K

| Término (inglés) | Español | Significado |
|---|---|---|
| Knowledge graph | grafo de conocimiento | Hechos guardados como tripletas entidad–relación–entidad. |
| KV cache | caché KV | Claves y valores de atención guardados para no recalcular los *tokens* anteriores al generar. |

## L

| Término (inglés) | Español | Significado |
|---|---|---|
| Latency | latencia | Tiempo desde la petición hasta la respuesta; en los LLM se suele dividir en tiempo hasta el primer *token* y tiempo por *token*. |
| LLM (large language model) | modelo de lenguaje grande (LLM) | Una red neuronal entrenada con grandes corpus de texto para predecir el siguiente *token*. |
| LLM-as-a-judge | LLM como juez | Usar un modelo para calificar salidas; útil pero sesgado, así que hay que calibrarlo. |
| Logits | *logits* | Las puntuaciones en bruto que produce un modelo para cada *token* antes del *softmax*. |
| LoRA (low-rank adaptation) | LoRA (adaptación de bajo rango) | *Fine-tuning* que aprende una pequeña actualización de bajo rango en lugar de cambiar todos los pesos. |
| Loss masking | enmascarado de la pérdida | Entrenar solo con algunos *tokens* (por ejemplo, la respuesta del asistente) en SFT. |

## M

| Término (inglés) | Español | Significado |
|---|---|---|
| MCP (Model Context Protocol) | MCP (Model Context Protocol) | Un protocolo abierto para conectar aplicaciones de IA con herramientas, recursos y *prompts* a través de servidores. |
| MinHash | MinHash | Una técnica para estimar rápido la similitud de Jaccard; se usa para detectar casi duplicados. |
| Mixture of Experts (MoE) | mezcla de expertos (MoE) | Una arquitectura en la que cada *token* activa solo algunas subredes expertas. |
| Multi-agent system | sistema multiagente | Varios agentes con roles distintos que cooperan en una tarea. |

## N

| Término (inglés) | Español | Significado |
|---|---|---|
| Near-duplicate | casi duplicado | Un documento casi idéntico a otro (pequeñas ediciones, distinto texto de plantilla). |
| Next-token prediction | predicción del siguiente *token* | El objetivo de entrenamiento de los modelos de lenguaje. |

## O

| Término (inglés) | Español | Significado |
|---|---|---|
| Orchestrator-worker | orquestador-trabajadores | Un *workflow* en el que un modelo divide una tarea y delega partes en llamadas trabajadoras. |
| Outcome reward model (ORM) | modelo de recompensa de resultado (ORM) | Un modelo que puntúa solo la respuesta final. |

## P

| Término (inglés) | Español | Significado |
|---|---|---|
| Parameter-efficient fine-tuning (PEFT) | ajuste fino eficiente en parámetros (PEFT) | Métodos que entrenan una pequeña fracción de los parámetros (adaptadores, LoRA). |
| Perplexity | perplejidad | La exponencial de la entropía cruzada media; cuanto más baja, menos se sorprende el modelo. |
| PII (personally identifiable information) | datos personales (PII) | Datos que identifican a una persona: nombres, correos, teléfonos, direcciones. |
| Post-training | post-entrenamiento | Todo lo que viene tras el preentrenamiento y convierte un modelo base en un asistente (SFT, RLHF, DPO). |
| PPO (Proximal Policy Optimization) | PPO (optimización de política proximal) | El algoritmo de aprendizaje por refuerzo del RLHF clásico. |
| Pre-training | preentrenamiento | Entrenar con enormes cantidades de texto para predecir el siguiente *token*. |
| Process reward model (PRM) | modelo de recompensa de proceso (PRM) | Un modelo que puntúa cada paso del razonamiento, no solo la respuesta. |
| Prompt | *prompt* | El texto de entrada que se da a un modelo. |
| Prompt chaining | encadenamiento de *prompts* | Dividir una tarea en una secuencia fija de llamadas al modelo. |
| Prompt injection | inyección de *prompts* | Texto que intenta anular las instrucciones del sistema, directamente u oculto en contenido recuperado. |

## Q

| Término (inglés) | Español | Significado |
|---|---|---|
| Quantization | cuantización | Guardar los pesos con menos bits (8, 4...) para ahorrar memoria y acelerar la inferencia. |

## R

| Término (inglés) | Español | Significado |
|---|---|---|
| RAFT | RAFT | *Retrieval-augmented fine-tuning*: entrenar un modelo para responder a partir de documentos con distractores. |
| RAG (retrieval-augmented generation) | generación aumentada por recuperación (RAG) | Recuperar documentos relevantes y dárselos al modelo como contexto. |
| ReAct | ReAct | Un bucle de agente que alterna razonamiento, acciones con herramientas y observaciones. |
| Reasoning model | modelo de razonamiento | Un modelo entrenado para pensar en cadenas largas antes de responder. |
| Recall@k | *recall@k* (exhaustividad en k) | La proporción de consultas cuyo elemento relevante aparece entre los k primeros resultados. |
| Reciprocal rank fusion (RRF) | fusión por rango recíproco (RRF) | Combinar listas ordenadas sumando 1 / (k + posición). |
| Reflection | reflexión | Un patrón en el que un modelo critica y mejora su propia salida. |
| Reranking | reordenación (*reranking*) | Volver a puntuar los primeros resultados recuperados con un modelo más preciso (y más lento). |
| Retrieval | recuperación | Encontrar los documentos más relevantes para una consulta. |
| ReWOO | ReWOO | Un patrón de agente que planifica primero todas las llamadas a herramientas, después las ejecuta y luego responde. |
| Reward hacking | explotación de la recompensa (*reward hacking*) | Un modelo que maximiza una recompensa de formas que no reflejan el objetivo buscado. |
| Reward model | modelo de recompensa | Un modelo entrenado con preferencias humanas para puntuar salidas. |
| RLHF | aprendizaje por refuerzo con retroalimentación humana (RLHF) | *Reinforcement learning from human feedback*. |
| Routing | enrutamiento | Clasificar una entrada y enviarla a un camino especializado. |

## S

| Término (inglés) | Español | Significado |
|---|---|---|
| Sampling | muestreo | Elegir el siguiente *token* al azar según sus probabilidades (ajustadas). |
| Self-attention | autoatención | Atención en la que las consultas, las claves y los valores salen de la misma secuencia. |
| Self-consistency | autoconsistencia | Muestrear varios caminos de razonamiento y quedarse con la respuesta más votada. |
| SFT (supervised fine-tuning) | ajuste fino supervisado (SFT) | Entrenar con ejemplos de entradas y salidas ideales. |
| Speculative decoding | decodificación especulativa | Un modelo borrador pequeño propone *tokens* que el modelo grande verifica en paralelo. |
| STaR (Self-Taught Reasoner) | STaR | Mejorar el razonamiento afinando el modelo con sus propias explicaciones correctas. |
| System prompt | *prompt* de sistema | Instrucciones que fijan el rol y las reglas del asistente en una conversación. |

## T

| Término (inglés) | Español | Significado |
|---|---|---|
| Temperature | temperatura | Un divisor de los *logits*: valores bajos hacen la salida más determinista y valores altos, más aleatoria. |
| Time to first token (TTFT) | tiempo hasta el primer *token* (TTFT) | La latencia hasta que llega el primer *token* de salida. |
| Token | *token* | La unidad de texto que lee y escribe un modelo: una palabra, parte de una palabra o un byte. |
| Tokenizer | tokenizador | El componente que convierte texto en *tokens* y viceversa. |
| Tool calling | llamada a herramientas | El modelo pide a la aplicación que ejecute una función con argumentos estructurados. |
| Top-k sampling | muestreo *top-k* | Muestrear solo entre los k *tokens* más probables. |
| Top-p (nucleus) sampling | muestreo *top-p* (de núcleo) | Muestrear entre el conjunto más pequeño de *tokens* cuyas probabilidades suman p. |
| Transformer | Transformer | La arquitectura de red neuronal detrás de los LLM modernos, basada en la atención. |
| Tree of Thoughts (ToT) | árbol de pensamientos (ToT) | Explorar varios caminos parciales de razonamiento como un árbol, con evaluación y poda. |

## V

| Término (inglés) | Español | Significado |
|---|---|---|
| Vector database | base de datos vectorial | Un almacén que indexa *embeddings* para buscar por similitud. |
| Verifier | verificador | Un programa o modelo que comprueba si una respuesta o un paso es correcto. |

## Z

| Término (inglés) | Español | Significado |
|---|---|---|
| Zero-shot prompting | *prompting* sin ejemplos (*zero-shot*) | Pedir al modelo una tarea solo con instrucciones, sin ejemplos. |

---

[🏠 Inicio del curso](../README.es.md) · [Recursos →](resources.es.md)
