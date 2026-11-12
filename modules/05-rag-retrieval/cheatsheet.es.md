🌐 [English](cheatsheet.md) · Español

# Chuleta del módulo 05 — RAG I: recuperación

## Definiciones

| Término | Significado |
|---|---|
| RAG (generación aumentada por recuperación, *retrieval-augmented generation*) | Buscar en documentos de confianza, poner los mejores pasajes en el *prompt* y responder a partir de ellos con citas |
| Flujo de indexación (fuera de línea) | Analizar → filtrar y deduplicar → fragmentar → indexar (palabras clave, vectores, hechos) |
| Flujo de consulta (en línea) | Pregunta → buscar → reordenar → construir el *prompt* → generar |
| Fragmento (*chunk*) | La unidad que indexas, recuperas y citas (`kb-007#4` = documento `kb-007`, trozo 4) |
| Índice invertido | término → apariciones (*postings*) {fragmento: frecuencia del término} |
| *Embedding* / bicodificador (*bi-encoder*) | Texto → vector; fragmentos codificados fuera de línea, preguntas en línea, con el mismo modelo |
| Índice basado en conocimiento | Hechos como tripletas (sujeto, predicado, objeto) con un documento de origen |
| ANN (vecino más cercano aproximado) | Cambia un poco de *recall* por mucho menos trabajo por consulta (IVF, HNSW, PQ) |
| Reordenación (*reranking*) | Reordenar un conjunto pequeño de candidatos con un puntuador más lento y más fuerte (un codificador cruzado en producción) |

## Fórmulas

```text
IDF(t)      = ln(1 + (N - df + 0.5) / (df + 0.5))
BM25(q, d)  = sum over t in q of IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
              k1 = 1.5 caps repetition, b = 0.75 normalises length
cosine(u,v) = u . v / (|u| |v|)        (= dot product for unit vectors)
RRF(item)   = sum over lists of 1 / (60 + rank)
recall@k    = share of questions with a relevant document in the top k
MRR         = mean of 1 / rank of the first relevant document (0 if none)
recall@10 (ANN) = share of the exact top 10 that the approximate index returns
```

## Tablas de decisión

| Necesidad | Elige |
|---|---|
| Comportamiento, tono, formato | *Prompting* o *fine-tuning* (módulo 04) |
| Corpus pequeño, estático y público | Prueba primero el contexto largo; consérvalo como línea base |
| Hechos que cambian, citas, acceso por persona usuaria | RAG |

| Fuente | Analizador |
|---|---|
| Markdown, HTML, JSON | Reglas (*front matter*, etiquetas meta, encabezados, tablas); descarta nav, footer y elementos ocultos |
| PDF digital | Modelo de maquetación + extracción de texto; valida los recuentos |
| Escaneos, tablas en imagen | OCR o un modelo de visión y lenguaje; valida y conserva los ids de página |

| Fragmentador | Úsalo cuando | Cuidado con |
|---|---|---|
| Fijo + solapamiento | Necesitas una línea base rápida | Corta tablas y frases |
| Recursivo | Prosa con párrafos | Ignora los encabezados |
| Por encabezados | Artículos de ayuda estructurados | Las secciones grandes y las tablas se convierten en un fragmento largo |
| Semántico | Prosa larga con cambios de tema | Solo es tan bueno como el *embedding* |
| Por filas de tabla | Tablas anchas | Muchos fragmentos pequeños de una misma página |

| Colección | Índice |
|---|---|
| Pequeña (Hearthwise hoy: 128 fragmentos) | Plano (*flat*), exacto |
| Grande, con memoria disponible | HNSW, ajusta `efSearch` |
| Grande, con memoria justa | IVF, ajusta `nprobe` |
| Demasiado grande para la memoria | IVF-PQ + reordenación exacta de los candidatos |

## Números para recordar (del lab)

| Resultado | Valor |
|---|---|
| Sin filtros: preguntas (de 40) con un doc sustituido o interno en el top 5 | 15 → 0 con filtros |
| Fragmentos por encabezado, *recall@5* de `table` | 0.00 → 1.00 con fragmentos por filas de tabla |
| *Recall@5* de BM25 / vectores por *hashing* / híbrido (fragmentos por filas de tabla) | 0.944 / 0.861 / 0.917 |
| Reordenador léxico: *recall@1* y MRR | 0.667 → 0.833 y 0.786 → 0.898 |
| Una pregunta de 36 | unos 0.028 de *recall* |
| IVF nprobe 1 / 8 / 32: *recall@10* | 0.741 / 0.917 / 0.977 |
| HNSW efSearch 16 / 64 / 128: *recall@10* | 0.819 / 0.951 / 0.981 |
| Memoria: plano / HNSW / IVF-PQ (50.000 × 64) | 12.80 / 26.41 / 1.33 MB |

## Comandos del lab

```bash
uv run lab parse                                  # structure, metadata, hidden elements
uv run lab chunk --chunker table --show kb-007    # filters, duplicates, chunks of one doc
uv run lab index                                  # sizes; timings are examples
uv run lab search "question" [--fold] [--rerank] [--retriever hybrid] [--no-filter]
uv run lab prompt "question"                      # what a generator would receive
uv run lab eval                                   # the lineup; add a flag for one config by category
uv run lab kg "Which Hearthwise device was launched most recently?"
uv run lab ann [--no-faiss] [--spread 3]          # exact vs IVF vs HNSW vs PQ
```

## Los tres errores más comunes

1. **Filtrar después de la recuperación.** Los documentos sustituidos e internos no deben llegar nunca
   al índice ni al *prompt*; un pasaje que está en el *prompt* se puede citar.
2. **Fiarse de una métrica sin leer los fragmentos.** Los fragmentos de tamaño fijo "encuentran" la
   tabla, pero pierden sus cabeceras; el *recall* de documentos no lo ve.
3. **Ajustar con el conjunto de evaluación y presentarlo como resultado.** Una puntuación perfecta en 36
   preguntas con las que ajustaste es un motivo para escribir preguntas nuevas.
