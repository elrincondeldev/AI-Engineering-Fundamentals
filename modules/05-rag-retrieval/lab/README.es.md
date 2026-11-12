🌐 [English](README.md) · Español

# Lab 05 — Recuperación para el centro de ayuda de Hearthwise

> Analiza, filtra, fragmenta e indexa el centro de ayuda de Hearthwise, mide cinco fragmentadores y tres
> recuperadores con las preguntas de evaluación compartidas, y compara la búsqueda exacta con la
> aproximada usando FAISS. ⏱️ unas 2 horas.

## 🎯 Qué vas a construir

La mitad de recuperación del chatbot de soporte que completa el módulo 06:

| Pieza | Archivo | Qué hace |
|---|---|---|
| Analizador (*parser*) | `src/m05_retrieval/parsing.py` | Convierte el *front matter* de Markdown y el HTML (etiquetas meta, encabezados, tablas, elementos ocultos) en bloques de `Document` |
| Fragmentadores y filtros | `chunking.py` | Fragmentadores fijo, recursivo, por encabezados, semántico y por filas de tabla; filtros de metadatos; eliminación de casi duplicados |
| Índice por palabras clave | `bm25.py` | Índice invertido y puntuación BM25 |
| Índice vectorial | `vectors.py` | Codificador por *hashing* sin conexión (o uno real a través de `providers.py`) y búsqueda exacta por coseno |
| Índice de conocimiento | `knowledge.py` | Hechos de producto como tripletas (sujeto, predicado, objeto) con su fuente |
| Fusión y reordenación | `hybrid.py`, `rerank.py` | Fusión por rango recíproco y un reordenador léxico de segunda etapa |
| Flujo y evaluación | `retrieval.py`, `evaluate.py` | Una sola `Config` construye todo el flujo; *recall@k*, MRR, ctx@5 y exposición sobre `rag_eval` |
| *Benchmark* ANN | `ann.py` | k-NN exacto en numpy, un IVF en numpy y los índices Flat, IVF, HNSW e IVF-PQ de FAISS sobre vectores generados con semilla |
| *Prompt* | `prompting.py` | El *prompt* fundamentado que recibiría un generador |

Todo funciona sin conexión en una CPU, sin clave de API. El mismo comando imprime siempre los mismos
números (solo cambian los tiempos).

## ⚙️ Preparación

Desde la raíz del repositorio:

```bash
cd modules/05-rag-retrieval/lab
uv sync
uv run pytest
```

Resumen esperado (el tiempo varía según la máquina):

```text
118 passed, 1 skipped in 6.70s
```

El test omitido habla con un servidor real de Ollama; consulta [🔌 Usar un modelo real](#-usar-un-modelo-real).
`uv run lab --help` lista los comandos: `parse`, `chunk`, `index`, `search`, `eval`, `prompt`, `kg`
y `ann`.

## 👣 Pasos

Cada paso tiene un comando, su salida esperada y una tarea **✍️ Constrúyelo**. El código de referencia
ya está en su sitio para que todo funcione. Para practicar, borra el cuerpo de la función indicada
(conserva su firma y su *docstring*), escribe tu propia versión y ejecuta el test indicado.
`git restore <file>` recupera la referencia si te atascas.

### Paso 1 — Analiza el centro de ayuda

```bash
uv run lab parse
```

La primera línea y las últimas de la salida:

```text
doc     format   lang status     audience headings texts tables hidden
kb-001  markdown en   current    public          6    13      1      0
kb-002  markdown en   current    public          6    11      1      0
...
kb-040  markdown en   current    public         13    34      2      0
kb-041  markdown en   current    internal        4     3      1      0

26 documents, 13 tables (80 data rows), 1 hidden HTML element(s) dropped
about 5,070 tokens of parsed text (rough count: words and punctuation)
```

Busca en la tabla `kb-010` (sustituido) y `kb-041` (interno), y `kb-025`, la página con un elemento
oculto.

**✍️ Constrúyelo:** reescribe `split_front_matter` y `parse_markdown_body` en `parsing.py`. Compruébalo
con `uv run pytest tests/test_parsing.py`.

### Paso 2 — Filtra y fragmenta

```bash
uv run lab chunk --chunker table --show kb-007
```

```text
near-duplicate: kb-027 ~ kb-026 (Jaccard 0.884)
dropped kb-010: status=superseded
dropped kb-027: near-duplicate of kb-026 (Jaccard 0.884)
dropped kb-041: audience=internal

chunker=table: 23 documents -> 128 chunks (words per chunk: mean 31.1, max 96)
wrote outputs/chunks-table.jsonl

--- kb-007#0 [Compare Hearthwise devices]
Compare Hearthwise devices
Use this table to compare the main specifications of every device we sell. Prices include VAT for Spain and Portugal.
Standby power is measured with the device connected to Wi-Fi and idle.

--- kb-007#1 [Compare Hearthwise devices]
Compare Hearthwise devices | Ember T3 | Price (EUR): 149; Wi-Fi: 2.4 GHz; Standby (W): 0.6; Needs hub: no; Weight (g): 210; Operating temperature (°C): 5 to 35; Launched: 2023-10-02
```

La salida continúa con un fragmento por dispositivo. Repite con `--chunker heading`, `fixed`,
`recursive` y `semantic`, y abre `outputs/chunks-<name>.jsonl` para compararlos.

**✍️ Constrúyelo:** reescribe `chunk_fixed` (ventanas con solapamiento) y `row_to_text` en
`chunking.py`. Compruébalo con `uv run pytest tests/test_chunking.py`.

### Paso 3 — Construye los índices

```bash
uv run lab index
```

```text
chunks:              128
chunk text:          22,048 bytes
vocabulary:          950 terms
postings:            3,361 (term, chunk) pairs
inverted index:      ~32,064 bytes (8 bytes per posting + the terms)
vector matrix:       131,072 bytes (128 x 256 float32)

build time (example, varies by machine): bm25 1.1 ms, vector 12.4 ms
query time (example, varies by machine): bm25 0.04 ms, vector 0.08 ms
```

**✍️ Constrúyelo:** reescribe `InvertedIndex.build`, `idf` y `BM25.term_score` en `bm25.py`.
Compruébalo con `uv run pytest tests/test_bm25.py`.

### Paso 4 — Busca y mira el prompt

```bash
uv run lab search "Who pays customs fees for orders shipped to Mexico?" --fold
```

```text
config: bm25/table/fold  (128 chunks)
1. kb-013#1  score=6.493  Spain and Portugal - Standard delivery takes 2 to 4 business days and costs €4.95. - Stand
2. kb-031#0  score=4.867  Envíos a Latinoamérica Enviamos a México, Colombia y Chile con una tarifa plana de **€14.9
3. kb-031#1  score=4.468  Plazos de entrega País Plazo México 7 a 10 días hábiles Colombia 8 a 12 días hábiles Chile
4. kb-026#3  score=4.069  Where can I download an invoice? In the app under Account > Orders. Tap the order and choo
5. kb-011#2  score=3.762  How to start a return 1. Open the Hearthwise app and go to Account > Orders. 2. Choose the
query time: 0.11 ms (example, varies by machine)
```

Ahora fusiona BM25 con los vectores por *hashing*:

```bash
uv run lab search "Who pays customs fees for orders shipped to Mexico?" --fold --retriever hybrid
```

```text
config: hybrid/table/fold  (128 chunks)
1. kb-026#3  score=0.032  Where can I download an invoice? In the app under Account > Orders. Tap the order and choo
2. kb-013#4  score=0.032  Tracking your order You get a tracking link by email when the parcel leaves our warehouse.
3. kb-013#1  score=0.030  Spain and Portugal - Standard delivery takes 2 to 4 business days and costs €4.95. - Stand
4. kb-003#4  score=0.029  Resetting to factory settings Hold the side button for 10 seconds until the LED blinks red
5. kb-006#4  score=0.029  Resetting to factory settings Hold the button on the back for 20 seconds until the ring li
query time: 0.20 ms (example, varies by machine)
```

La página en español ha desaparecido: la lección 5.3 explica por qué. Por último, mira lo que recibiría
un generador: `uv run lab prompt "Can I return an opened Glow A19?"` (la salida está en la lección 5.1).

**✍️ Constrúyelo:** reescribe `rrf_fuse` en `hybrid.py` y `format_context` en `prompting.py`.
Compruébalo con `uv run pytest tests/test_vectors_hybrid_rerank.py tests/test_prompting_and_providers.py`.

### Paso 5 — Evalúa fragmentadores y recuperadores

```bash
uv run lab eval
```

```text
36 answerable questions. R@k, MRR: document ranking. ctx@5: first 5 chunks.
table, x-ling: R@5 per category. exposed: questions (of 40) with a superseded or internal doc in the top 5.

config                       chunks   R@1   R@5   MRR ctx@5 table x-ling exposed
bm25/heading/nofilter           119 0.611 0.917 0.745 0.917  0.00   0.50      15
bm25/heading                    111 0.667 0.889 0.770 0.861  0.00   0.50       0
bm25/fixed                       65 0.750 0.972 0.840 0.944  1.00   0.50       0
bm25/recursive                   52 0.778 0.972 0.863 0.972  1.00   0.50       0
bm25/semantic                    98 0.639 0.917 0.768 0.917  0.00   0.50       0
bm25/table                      128 0.667 0.944 0.786 0.917  1.00   0.50       0
vector/table                    128 0.722 0.861 0.804 0.861  0.00   0.00       0
hybrid/table                    128 0.750 0.917 0.816 0.889  0.50   0.00       0
bm25/table/fold                 128 0.667 0.972 0.800 0.944  1.00   1.00       0
bm25/table/rerank               128 0.833 0.972 0.898 0.944  1.00   0.50       0
bm25/table/fold/rerank          128 0.833 1.000 0.905 0.972  1.00   1.00       0
```

La primera fila coincide con la línea base de referencia de
[`datasets/README.es.md`](../../../datasets/README.es.md) en el *recall@5* (0.917) y en las categorías
`table` y `cross_lingual` (0.00 y 0.50); el lab divide las páginas HTML por encabezado, así que tiene
119 fragmentos en lugar de los 118 de la línea base y un MRR de 0.745 en lugar de 0.754. Cualquier
opción (`--chunker`, `--retriever`, `--no-filter`, `--fold`, `--rerank`, `--title`) cambia a una sola
configuración con el *recall@5* por categoría y la lista de preguntas falladas.

**✍️ Constrúyelo:** reescribe `doc_ranking`, `recall_at_k` y `reciprocal_rank` en `evaluate.py`.
Compruébalo con `uv run pytest tests/test_evaluate.py tests/test_documented_numbers.py`.

### Paso 6 — Pregunta al grafo de conocimiento

```bash
uv run lab kg "Which Hearthwise device was launched most recently?"
```

```text
90 triples about 6 products
(Lookout C2, Launched, 2025-09-01)  source: kb-005
```

Después pregunta `uv run lab kg "What is the price of the Lookout C2 in Mexico?"`. Responde €89 desde
`kb-005` y 89 desde `kb-007`, aunque el centro de ayuda nunca indica un precio en México.

**✍️ Constrúyelo:** `uv run lab kg "What is the SKU of the Watcher S1?"` no encuentra nada, aunque todas
las páginas de producto indican su SKU. Añade una pista de predicado a `PREDICATE_HINTS` en
`knowledge.py` para que devuelva `(Watcher S1, SKU, HW-WCH-S1)`, y añade un test para ello en
`tests/test_knowledge.py`.

### Paso 7 — Búsqueda exacta frente a aproximada

```bash
uv run lab ann
```

```text
50,000 vectors x 64 dims, 200 queries, recall@10 against exact search

method               params                    recall@10  compared memory MB ms/query*
numpy exact          -                             1.000    50,000     12.80     0.472
numpy IVF            nlist=256 nprobe=1            0.750       484     13.27     0.035
numpy IVF            nlist=256 nprobe=8            0.917     1,825     13.27     0.179
faiss IndexFlatIP    -                             1.000    50,000     12.80     0.233
faiss IndexIVFFlat   nlist=256 nprobe=1            0.741       474     13.27     0.004
faiss IndexIVFFlat   nlist=256 nprobe=8            0.917     1,814     13.27     0.025
faiss IndexIVFFlat   nlist=256 nprobe=32           0.977     6,504     13.27     0.048
faiss IndexHNSWFlat  M=32 efSearch=16              0.819       649     26.41     0.015
faiss IndexHNSWFlat  M=32 efSearch=64              0.951     1,681     26.41     0.053
faiss IndexHNSWFlat  M=32 efSearch=128             0.981     3,055     26.41     0.112
faiss IndexIVFPQ     nlist=256 m=16 nprobe=8       0.593     1,814      1.33     0.017

* example timings, vary by machine. compared = similarities computed per query
```

Tarda unos segundos. En otra CPU, el *recall* puede variar en el último dígito; los tests admiten ±0.02.

**✍️ Constrúyelo:** reescribe `exact_knn` y `NumpyIVF.search` en `ann.py`. Compruébalo con
`uv run pytest tests/test_ann.py`, y después ejecuta `uv run lab ann --no-faiss` y compara tu IVF con
el de FAISS.

## 💥 Rómpelo

**1. Desactiva los filtros.** Ejecuta `uv run lab eval --chunker table --no-filter`. ¿Qué les pasa al
*recall@5*, al MRR y a la columna `exposed`?

<details><summary>Qué deberías ver</summary>

El *recall@5* se queda en 0.944, el MRR baja de 0.786 a 0.758 y `exposed` salta de 0 a 15. El *recall*
por sí solo te diría que no pasa nada. La política antigua y el manual interno compiten ahora por el top
5, y en q039 la sección interna sobre créditos de cortesía es el primer resultado.

</details>

**2. Quita la normalización por longitud.** En `bm25.py`, cambia el valor por defecto `b: float = 0.75` por `b: float = 0.0`
y ejecuta `uv run lab eval --chunker heading`.

<details><summary>Qué deberías ver</summary>

El *recall@5* sube de 0.889 a 0.944, el MRR de 0.770 a 0.843, y la categoría `table` pasa de 0.00 a
1.00: el fragmento largo de la tabla comparativa ya no se penaliza. Pero ahora se falla q021 (entrega
en Chile). `b` es un parámetro ajustable, no una ley; moverlo cambia qué preguntas ganan. Después,
vuelve a poner `0.75`.

</details>

**3. Tokeniza las consultas de forma distinta a los documentos.** En `BM25.scores`, sustituye
`tokenize(query, fold=self.fold)` por `query.split()` y ejecuta `uv run lab eval --chunker table`.

<details><summary>Qué deberías ver</summary>

El *recall@5* cae de 0.944 a 0.861, el MRR de 0.786 a 0.739, y se fallan 5 preguntas: "Watcher" ya no
coincide con "watcher", y "weigh?" (con su signo de interrogación) no coincide con nada. Los documentos
y las consultas tienen que pasar por el mismo tokenizador.

</details>

**4. Destruye los clústeres.** Ejecuta `uv run lab ann --no-faiss --spread 3`. Ahora los vectores
están mucho menos agrupados. ¿Qué le pasa al IVF en numpy?

<details><summary>Qué deberías ver</summary>

El *recall@10* se desploma a 0.069 con nprobe = 1 y a 0.301 con nprobe = 8 (frente a 0.750 y 0.917 con
la dispersión por defecto). IVF supone que los vecinos cercanos comparten clúster; cuando los datos no
tienen estructura de clústeres, explorar unas pocas listas encuentra pocos vecinos. Los *embeddings*
reales están agrupados, pero cuánto es algo que se mide, no que se supone.

</details>

## 🚀 Amplíalo

**1. Cabeceras de contexto en los fragmentos.** Ejecuta `uv run lab eval --chunker table --title`, que
antepone a cada fragmento el título de su documento. Después amplía `with_title` en `chunking.py` para
añadir también el encabezado de la sección (`Title > Section`) y vuelve a medir.

<details><summary>Pista y resultado de referencia</summary>

Con el título como prefijo, el *recall@1* pasa de 0.667 a 0.722, el *recall@5* de 0.944 a 0.972 y el
MRR de 0.786 a 0.824; solo se falla q022. El título lleva palabras como "Returns policy" o "Factory
reset" a cada fragmento de su página, que es justo el contexto que pierde un fragmento por encabezado.
Para la sección, usa `Chunk.section`, que todos los fragmentadores ya rellenan.

</details>

**2. Expansión bilingüe de consultas.** Haz que q022 funcione sin normalización de acentos: añade un
pequeño glosario de inglés a español (customs, shipping, delivery, returns, warranty) y añade las
traducciones a la consulta antes de buscar.

<details><summary>Pista</summary>

Escribe `expand_query(q: str) -> str` en un `query.py` nuevo, aplícala en `BM25Retriever.search` y
exponla como una opción de `Config` para que `uv run lab eval` pueda compararla. Comprueba las columnas
`cross_lingual` y `exposed`: la expansión no debe arrastrar páginas sin relación. Un sistema real
traduciría la consulta con un LLM o usaría un modelo de *embeddings* multilingüe; el glosario muestra
la idea sin conexión.

</details>

**3. Reducción ligera a la raíz (*stemming*).** Enseña al tokenizador que "weigh" y "weight", o
"device" y "devices", son el mismo término.

<details><summary>Pista</summary>

Añade una opción `stem` a `tokenize` en `text.py` que quite unos pocos sufijos ingleses (`-s`, `-es`,
`-t` después de `weigh`, `-ing`), úsala en los dos lados en `InvertedIndex.build` y `BM25.scores`, y
mide. Vigila q004 ("devices" en la pregunta de la garantía) y q017 (weigh). Después comprueba que las
preguntas en español no empeoraron.

</details>

**4. Búsqueda comprimida con una segunda pasada exacta.** Envuelve el índice IVF-PQ en
`faiss.IndexRefineFlat` para que los candidatos encontrados con códigos comprimidos se vuelvan a
puntuar con los vectores completos.

<details><summary>Pista</summary>

En `_faiss_suite` de `ann.py`: `refine = faiss.IndexRefineFlat(ivfpq)`, después `refine.k_factor = 4`
y `refine.search(queries, k)`. Informa del *recall@10* y de la memoria (el índice de refinado también
guarda los vectores en bruto, así que la memoria sube). Compara con IVF-Flat simple con el mismo
`nprobe`.

</details>

**5. Un modelo de *embeddings* real.** Ejecuta las configuraciones vectorial e híbrida con *embeddings*
de Ollama (consulta la sección siguiente) y compara las columnas `cross_lingual` y `table` con las del
codificador por *hashing*.

<details><summary>Pista</summary>

`--embedder provider` ya está conectado en `search`, `eval` y `prompt`. Elige un modelo de *embeddings*
multilingüe si quieres que la página de envíos en español coincida con preguntas en inglés, y apunta el
nombre del modelo junto a tus números.

</details>

## ✅ Definición de terminado (*Definition of Done*)

- [ ] `uv run pytest` pasa (118 passed, 1 skipped).
- [ ] Reescribiste al menos cuatro de las funciones de **Constrúyelo** y sus tests pasan.
- [ ] Sabes explicar, con el texto de los fragmentos, por qué `bm25/heading` falla q017 y q018 y
      `bm25/table` no.
- [ ] Ejecutaste todos los experimentos de **Rómpelo** y sabes decir qué enseña cada uno.
- [ ] Sabes explicar por qué `hybrid/table` puntúa menos que `bm25/table` con estos datos.
- [ ] Sabes leer la tabla de `ann` y recomendar un índice para 128 fragmentos y para 50.000 vectores.
- [ ] Restauraste todo el código que cambiaste para **Rómpelo** (`git restore` o a mano).

## 🧯 Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| `Could not find the repo's datasets/ folder` | El lab se copió fuera del repositorio | Define `COURSE_DATASETS_DIR=/path/to/datasets` |
| `uv sync` no encuentra un *wheel* de `faiss-cpu` | Plataforma o versión de Python poco habituales | Usa Python 3.12 (`uv python install 3.12`). Si sigue fallando, quita `faiss-cpu` de `pyproject.toml`: las partes en numpy de `uv run lab ann --no-faiss` siguen funcionando y las filas de FAISS de los tests se omiten |
| El *recall* ANN difiere de la tabla en el último dígito | Instrucciones de CPU o versión de FAISS distintas | Es lo esperado; los tests admiten ±0.02 |
| `Network access is disabled in tests` | Un test intentó conectarse a un servidor | Los tests funcionan sin conexión por diseño; marca los tests con modelos reales como `ollama` u `openai` |
| `ProviderError: Cannot reach http://localhost:11434` | `--embedder provider` con `LLM_PROVIDER=ollama` y sin servidor | Arranca Ollama (`ollama serve`) o quita `--embedder provider` |
| Los números de `eval` cambiaron | Editaste un tokenizador, un fragmentador o un parámetro de BM25 | `git diff src/` muestra qué; `tests/test_documented_numbers.py` indica qué número se movió |

## 🔌 Usar un modelo real

Los vectores del lab salen del codificador por *hashing* sin conexión de `providers.py`, que relaciona
ortografía, no significado. Para usar un modelo de *embeddings* real, elige un proveedor y pasa
`--embedder provider`; todo lo demás sigue igual.

**Ollama (local, gratis):**

```bash
ollama pull nomic-embed-text              # the default embedding model in providers.py
LLM_PROVIDER=ollama OLLAMA_EMBED_MODEL=nomic-embed-text \
  uv run lab eval --retriever hybrid --chunker table --embedder provider
```

**Cualquier *endpoint* compatible con OpenAI (vLLM, LM Studio, servidor de llama.cpp, una API alojada):**

```bash
LLM_PROVIDER=openai OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_MODEL=<chat-model> \
  OPENAI_EMBED_MODEL=<embedding-model> OPENAI_API_KEY=<key-if-needed> \
  uv run lab eval --retriever vector --chunker table --embedder provider
```

El proveedor exige `OPENAI_MODEL` aunque aquí solo se usen *embeddings*.

**Qué no se probó.** Estas rutas con modelos reales no se ejecutaron para este módulo: Ollama estaba
instalado en la máquina de autoría, pero no en marcha, y no se usó ninguna clave de API. Todos los
números de las lecciones, de este README y del quiz salen del codificador por *hashing* sin conexión. El
test con modelo real
`tests/test_prompting_and_providers.py::test_ollama_embeddings_rank_the_returns_policy` solo se ejecuta
con `RUN_LIVE_TESTS=1 uv run pytest -m ollama` y un servidor en marcha. Si lo ejecutas, apunta tus
números con el nombre del modelo: son tuyos, no del curso.

---

[← Anterior: 5.4 Búsqueda exacta y aproximada](../04-exact-and-approximate-search.es.md) · [🏠 Módulo 05](../README.es.md) · [Siguiente: Quiz →](../quiz.es.md)
