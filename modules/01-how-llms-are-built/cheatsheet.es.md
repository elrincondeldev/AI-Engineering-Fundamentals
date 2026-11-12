🌐 [English](cheatsheet.md) · Español

# Chuleta del módulo 01 — Cómo se construyen los LLM

## El camino

```text
rastreo en bruto → limpieza → tokenizador → preentrenamiento → modelo base → post-entrenamiento → asistente → inferencia en una aplicación
```

## Definiciones clave

| Término | Significado |
|---|---|
| *Token* | Un trozo de texto (byte, subpalabra, palabra, signo de puntuación) con un *id* entero |
| Predicción del siguiente *token* | El modelo da una probabilidad para cada *token* del vocabulario; la generación repite un *token* cada vez |
| Modelo base | Resultado del preentrenamiento; continúa texto |
| Asistente | Modelo base tras el post-entrenamiento (instrucciones, preferencias); responde y sigue formatos |
| WARC / WAT / WET | Formatos de Common Crawl: respuestas en bruto / metadatos / texto plano extraído |
| Tejas (*shingles*) | Los *n-gramas* de palabras de un documento (aquí n = 5) |
| Similitud de Jaccard | \|A ∩ B\| / \|A ∪ B\| de dos conjuntos de tejas |
| MinHash | Firma cuyas posiciones coincidentes estiman la similitud de Jaccard |
| Bandas de LSH | Comparar solo los documentos que comparten una banda entera de la firma |
| Descontaminación | Eliminar el texto de entrenamiento que se solapa con los conjuntos de evaluación |
| BPE | Partir de 256 bytes; fusionar una y otra vez el par adyacente más frecuente |
| *Embedding* | Vector aprendido para cada *id* de *token* |
| Q, K, V | Consulta (lo que busco), clave (lo que ofrezco), valor (lo que entrego) |
| Máscara causal | La posición i solo puede atender a las posiciones j ≤ i |
| RoPE | Rota Q y K según la posición para que las puntuaciones dependan de la distancia relativa |
| MoE | Un enrutador envía cada *token* a unos pocos expertos; parámetros totales frente a activos |
| GQA | Varias cabezas de consulta comparten una cabeza de clave y valor; caché KV más pequeña |

## Fórmulas

| Qué | Fórmula |
|---|---|
| Precisión / *recall* | TP / (TP + FP) · TP / (TP + FN) |
| Umbral del 50 % de LSH | (1 / bandas) ^ (1 / filas); 16 bandas × 8 filas ≈ 0.707 |
| Atención | softmax(Q Kᵀ / √d_head + máscara) V |
| Entropía cruzada | − ln p(siguiente *token* correcto); suposición uniforme = ln V (ln 1024 = 6.931) |
| Parámetros por capa (clásico) | atención 4D² + *feed-forward* 2DF; sin *embeddings* ≈ 12 L D² cuando F = 4D |
| *Embeddings* | V × D (más otra vez V × D si la capa de salida no está atada) |
| Memoria de los pesos | parámetros × bytes (fp32 4, bf16 2, int8 1, int4 0,5) |
| Cómputo de entrenamiento | ≈ 6 × parámetros × *tokens* FLOPs |
| Caché KV | 2 × capas × kv_heads × d_head × bytes, por *token* |
| MoE | total = parte compartida + todos los expertos; activos = parte compartida + expertos elegidos |

## Números del módulo para recordar (del lab)

| Resultado | Valor |
|---|---|
| Rastreo → páginas conservadas | 134 → 82 (etiquetas: 83) |
| Precisión / *recall* del *pipeline* | 1.000 / 0.988; con `--min-words 50`: *recall* 0.687 |
| Caracteres de HTML conservados como texto | 25,4 % |
| Casi duplicados frente a artículos distintos (Jaccard) | alrededor de 0.97 frente a como mucho 0.417 |
| *Tokens* por palabra, en / es (vocabulario 1024) | 2.932 / 3.345 (ratio 1.141); 1.080 con el español ×5 |
| Pérdida de validación del bigrama neuronal | 6.932 → 2.985 (bigrama de conteo 3.065) |
| Parámetros de `classic-small` | 124.439.808 (*embeddings* 31,0 %) |
| Totales / activos de `moe-8x-top2` | 4.728.358.912 / 1.406.470.144 |

## Tablas de decisión

| Si necesitas... | Elige |
|---|---|
| Detectar copias idénticas de forma barata | Normalización de URL + *hash* exacto del texto |
| Detectar copias editadas a escala | MinHash + LSH |
| Conservar páginas útiles con datos personales | Anonimizar, no descartar |
| Secuencias más cortas en un idioma | Más datos de ese idioma para entrenar el tokenizador, o un vocabulario más grande |
| Planificar la memoria de un modelo MoE | Parámetros totales + caché KV |
| Planificar la latencia de un modelo MoE | Parámetros activos |
| Mantener los datos en tus máquinas | Pesos abiertos, alojados por ti |

## Comandos del lab

```bash
uv run lab ngram          # next-word probabilities, backoff, n-gram coverage
uv run lab ask            # base-style continuation vs assistant (LLM_PROVIDER)
uv run lab clean          # cleaning pipeline + precision/recall (--min-words, --skip)
uv run lab minhash        # exact vs MinHash Jaccard
uv run lab bpe            # train BPE, tokens per word en/es (--vocab-size, --es-weight, --text)
uv run lab attention      # shapes + causal attention weights
uv run lab train          # gradient descent (--lr, --steps)
uv run lab params         # parameters, memory, KV cache (--show)
uv run lab facts          # published figures with sources
```

## Los tres errores más comunes

1. **Tomar una salida fluida por un hecho.** La predicción del siguiente *token* produce texto
   probable; fundamenta y verifica.
2. **Copiar el umbral de un filtro sin medirlo.** Un mínimo de 50 palabras bajó el *recall* de 0.988 a
   0.687 en este rastreo.
3. **Enmascarar la atención después del *softmax*.** Enmascara las puntuaciones con menos infinito
   antes del *softmax*.

---

[← Anterior: Quiz](quiz.es.md) · [🏠 Módulo 01](README.es.md) · [Siguiente: Módulo 02 →](../02-decoding-post-training-evaluation/README.es.md)
