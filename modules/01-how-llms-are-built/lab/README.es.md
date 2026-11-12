🌐 [English](README.md) · Español

# Lab 01 — De un rastreo web en bruto a un bloque Transformer

> Limpia el rastreo web de Hearthwise y mídelo frente a las etiquetas, entrena un tokenizador BPE a
> nivel de byte, ejecuta la autoatención causal y el descenso de gradiente en numpy, y dimensiona
> familias de modelos reales. ⏱️ unas 3 horas.

## 🎯 Qué vas a construir

| Comando | Qué construyes | Lección |
|---|---|---|
| `uv run lab ngram`, `uv run lab ask` | Un predictor de la siguiente palabra por conteo; continuación de modelo base frente a asistente | [1.1](../01-llm-overview.es.md) |
| `uv run lab clean`, `uv run lab minhash` | Un *pipeline* de limpieza para preentrenamiento puntuado con precisión y *recall*; detección de casi duplicados con MinHash | [1.2](../02-pretraining-data.es.md) |
| `uv run lab bpe` | Un tokenizador BPE a nivel de byte, y el coste del español en *tokens* | [1.3](../03-tokenization.es.md) |
| `uv run lab attention`, `uv run lab train` | Autoatención causal multicabeza, un bloque decodificador, RoPE y descenso de gradiente | [1.4](../04-neural-networks-and-transformers.es.md) |
| `uv run lab params`, `uv run lab facts` | Una calculadora de parámetros, memoria, cómputo y caché KV, para modelos densos y MoE | [1.5](../05-model-families.es.md) |

Todo funciona sin conexión en una CPU en cuestión de segundos. La única dependencia además de la
biblioteca estándar es `numpy`. El código está en `src/m01_foundations/`:

| Archivo | Qué contiene |
|---|---|
| `ngram.py` | `NGramModel` con retroceso (*backoff*), `coverage()` |
| `cleaning.py` | `normalise_url`, `html_to_text`, `detect_language`, `quality_signals`, `redact_pii`, descontaminación, `run_pipeline`, `evaluate` |
| `dedup.py` | `content_hash`, `shingles`, `MinHasher`, `lsh_candidates`, `UnionFind` |
| `bpe.py` | `pretokenize`, `pair_counts`, `merge_pair`, `train_bpe`, `BPETokenizer` |
| `transformer.py` | `softmax`, `layer_norm`, `causal_mask`, `rope`, `scaled_dot_product_attention`, `decoder_block`, `forward`, `cross_entropy` |
| `neural.py` | `NeuralBigram` con un gradiente escrito a mano, `train_neural_bigram` |
| `params.py` | `ModelShape`, `count_params`, `weight_memory_gb`, `kv_cache_bytes`, `training_flops`, configuraciones predefinidas |
| `published.py` | Cifras citadas de artículos, cada una con la URL de su fuente |
| `providers.py`, `data.py` | La capa de proveedores compartida del curso y el cargador de datasets (copiados sin cambios) |

**Cómo usar este lab.** Cada paso tiene una tarea 🛠️ **Constrúyelo**. La implementación de referencia
ya está en el archivo, así que los comandos funcionan desde el principio. Para aprenderlo, borra el
cuerpo de la función indicada (conserva la firma y el *docstring*), escribe tu propia versión y ejecuta
el archivo de tests indicado hasta que pase. Si te atascas, `git restore <file>` recupera la
referencia.

## ⚙️ Preparación

Necesitas [uv](https://docs.astral.sh/uv/). Instala Python 3.12 por ti.

```bash
cd modules/01-how-llms-are-built/lab
uv sync
uv run pytest
```

Resumen esperado (el tiempo varía según la máquina):

```text
110 passed
```

`uv run lab --help` lista todos los comandos.

## 👣 Pasos

### Paso 1 — Prepara el entorno y ejecuta los tests

Haz la preparación de arriba y después lista los comandos:

```bash
uv run lab --help
```

```text
usage: lab [-h] {ngram,ask,clean,minhash,bpe,attention,train,params,facts} ...

Module 01 lab: how LLMs are built.
```

Abre `tests/test_cleaning.py` y lee dos tests. Los tests son la especificación contra la que
programarás en cada tarea de Constrúyelo.

### Paso 2 — Predice la siguiente palabra

```bash
uv run lab ngram
```

```text
Trained a 4-gram counting model on 77 cleaned English pages.

Prompt: 'Lower the setpoint by one degree in the'  (matched the last 3 word(s))
  living       0.364
  bedroom      0.273
  bathroom     0.091
  hallway      0.091
  home         0.091
  greedy continuation: living room and see if anyone notices . a thermostat near the

Prompt: 'The Ember T3'  (matched the last 3 word(s))
  costs        1.000
  greedy continuation: costs € 149 and launched on 2023 - 05 - 22 .

Held-out n-grams already seen in training (61 train / 16 held-out pages):
  n=1: 0.973
  n=2: 0.855
  n=3: 0.735
  n=4: 0.645
  n=5: 0.576
```

La fecha de lanzamiento es incorrecta (el Ember T3 salió a la venta el 2023-10-02). La lección 1.1
explica por qué. Después compara una continuación de estilo base con la respuesta de un asistente:

```bash
uv run lab ask
```

```text
Question: How do I reset my thermostat?

Base-model style (4-gram continuation): ...user loopback : i have a 6 m hallway and the sensor misses people at
Assistant via LLM_PROVIDER=fake (fake-1): [fake:81d8b57f] You said: How do I reset my thermostat?
```

🛠️ **Constrúyelo:** reescribe `NGramModel.next_word_probs` en `ngram.py` (primero el contexto más
largo y después retrocede). Compruébalo con `uv run pytest tests/test_ngram.py`.

### Paso 3 — Limpia el rastreo

```bash
uv run lab clean
```

```text
Crawl: 134 pages (128 with HTTP 200, 6 errors; 131 text/html, 3 other types). Labels say keep 83.

Dropped by stage:
  fetch          -9   -> 125
  url_dedup      -4   -> 121
  url_filter     -0   -> 121
  extract        -0   -> 121
  language       -8   -> 113
  quality        -13  -> 100
  exact_dup      -8   -> 92
  near_dup       -8   -> 84
  contamination  -2   -> 82

Kept 82 pages: 48,932 characters of text from 192,610 characters of raw HTML (25.4%).
Languages kept: en=77, es=5
PII redacted on 4 pages: ADDRESS=1, EMAIL=2, PHONE=2

Against labels.jsonl (positive class = keep):
  TP=82  FP=0  FN=1  TN=51
  precision=1.000  recall=0.988  F1=0.994
  dropped but should keep: https://corvid-home.example/thermo-x  (quality:repetitive)

Wrote outputs/clean_corpus.jsonl
```

Ahora mide dos variantes y rellena tú la tabla de la lección 1.2:

```bash
uv run lab clean --min-words 50
uv run lab clean --skip near_dup
```

Líneas clave esperadas: `precision=1.000  recall=0.687  F1=0.814` y
`precision=0.911  recall=0.988  F1=0.948`.

🛠️ **Constrúyelo:** reescribe `normalise_url`, `quality_signals` y `redact_pii` en `cleaning.py`.
Compruébalo con `uv run pytest tests/test_cleaning.py`, después vuelve a ejecutar `uv run lab clean` y
confirma que salen los mismos números.

### Paso 4 — Mide los casi duplicados con MinHash

```bash
uv run lab minhash
```

```text
MinHash with 128 hash functions; LSH with 16 bands of 8 rows (50% candidate threshold ~ 0.707)

pair                                                        exact J  MinHash
lighting-lighting-scenes-that-actually-get-used               0.969    0.977
lighting-smart-lighting-ideas-for-every-room-guide            0.970    0.969
standby-hidden-energy-what-your-gadgets-use-when-idle-gu      0.969    0.984
lighting-choosing-the-right-colour-temperature-guide          0.968    0.992
lighting-smart-lighting-ideas-for-every-room                  0.969    0.977
standby-a-weekend-project-to-find-vampire-loads-guide         0.973    0.969
standby-how-to-cut-standby-power-at-home                      0.969    0.992
standby-a-weekend-project-to-find-vampire-loads               0.975    0.953

Most similar pair of different articles: J=0.417
  https://techtips.example/heating/comfort-without-waste-in-winter-guide
  https://homeideas.example/heating/comfort-without-waste-in-winter
```

Prueba `uv run lab minhash --num-perm 32`: las estimaciones se vuelven más ruidosas. Menos funciones
*hash* significan una estimación más barata pero menos precisa.

🛠️ **Constrúyelo:** reescribe `MinHasher.signature` y `lsh_candidates` en `dedup.py`. Compruébalo con
`uv run pytest tests/test_dedup.py`.

### Paso 5 — Entrena un tokenizador BPE

```bash
uv run lab bpe
```

```text
Trained byte-level BPE on 82 documents (48,999 bytes): 767 merges + 256 bytes + 1 special = vocab 1024

First 10 merges:
  #0        '·' + 't'      -> ' t'       (seen 1003 times)
  #1        'h' + 'e'      -> 'he'       (seen 955 times)
  #2        '·' + 'a'      -> ' a'       (seen 900 times)
  #3        '·' + 's'      -> ' s'       (seen 797 times)
  #4        'i' + 'n'      -> 'in'       (seen 672 times)
  #5       '·t' + 'he'     -> ' the'     (seen 657 times)
  #6        'e' + 'r'      -> 'er'       (seen 623 times)
  #7        'o' + 'n'      -> 'on'       (seen 525 times)
  #8        'o' + 'r'      -> 'or'       (seen 506 times)
  #9        '·' + 'c'      -> ' c'       (seen 416 times)

'How long does shipping take?'
  12 tokens: H | ow | ·lon | g | ·does | ·sh | i | pping | ·t | a | ke | ?

'¿Cuánto tarda el envío?'
  19 tokens: <C2> | <BF> | C | u | <C3> | <A1> | nt | o | ·t | ar | d | a | ·el | ·en | v | <C3> | <AD> | o | ?

Special token: 'Hola<|endoftext|>Hello' -> [72, 316, 97, 1023, 72, 744, 111]

Tokens per word on Hearthwise tickets (846 en / 378 es):
   vocab      en      es   es/en
     257   5.566   5.726   1.029
     384   3.864   4.193   1.085
     512   3.519   3.907   1.110
     768   3.139   3.565   1.136
    1024   2.932   3.345   1.141
```

Después reequilibra los datos de entrenamiento: `uv run lab bpe --es-weight 5`. La última fila pasa a
ser `1024   2.993   3.231   1.080`.

🛠️ **Constrúyelo:** reescribe `merge_pair`, `train_bpe` y `BPETokenizer._encode_chunk` en `bpe.py`.
Compruébalo con `uv run pytest tests/test_bpe.py`. Los tests de ida y vuelta con emojis y texto en
chino son los que atrapan la mayoría de los errores.

### Paso 6 — Ejecuta la autoatención causal

```bash
uv run lab attention
```

```text
Text: 'Lower the setpoint by one degree'
Tokens (7): L | ower | ·the | ·setpoint | ·by | ·one | ·degree

Config: vocab=1024 d_model=32 heads=4 d_head=8 layers=2 d_ff=128
  token ids          (7,)
  embeddings x       (7, 32)
  queries per head   (4, 7, 8)   (heads, tokens, d_head)
  scores Q.K^T       (4, 7, 7)
  attention weights  (4, 7, 7)
  logits             (7, 1024)   (one score per vocabulary entry, per position)

Layer 0, head 0 attention weights (row = query token, column = key token):
                L   ower   ·the ·setpo    ·by   ·one ·degre
L            1.00   0.00   0.00   0.00   0.00   0.00   0.00
ower         0.50   0.50   0.00   0.00   0.00   0.00   0.00
·the         0.34   0.34   0.33   0.00   0.00   0.00   0.00
·setpoint    0.24   0.25   0.26   0.25   0.00   0.00   0.00
·by          0.20   0.20   0.20   0.20   0.20   0.00   0.00
·one         0.16   0.16   0.17   0.17   0.17   0.17   0.00
·degree      0.14   0.14   0.14   0.14   0.15   0.15   0.14

Every row sums to 1: True; weights above the diagonal: 0.0
Next-token loss with random weights: 6.905 (uniform guess: ln 1024 = 6.931)
```

🛠️ **Constrúyelo:** reescribe `scaled_dot_product_attention`, `rope` y `decoder_block` en
`transformer.py`. Compruébalo con `uv run pytest tests/test_transformer.py`. Los tests demuestran la
causalidad de principio a fin y la propiedad de posición relativa de RoPE.

### Paso 7 — Entrena con descenso de gradiente

```bash
uv run lab train
```

```text
Neural bigram: embedding 1024x32 -> logits 32x1024; 11,204 train / 2,596 validation tokens; lr=10.0

  step    train    valid
     0    6.932    6.932
   200    4.237    4.600
   400    2.832    3.376
   600    2.297    3.036
   800    2.042    2.986
  1000    1.914    2.985

Uniform guess:            6.931
Counting bigram (smoothed): 3.065
```

🛠️ **Constrúyelo:** reescribe `NeuralBigram.step` en `neural.py`. `test_gradient_matches_finite_differences`
compara tu gradiente con una estimación numérica; es la forma habitual de depurar una pasada hacia
atrás.

### Paso 8 — Cuenta parámetros, densos y MoE

```bash
uv run lab params
```

```text
preset                         total          active  bf16 GB  KV MB/1k tok
lab-tiny                      57,984          57,984     0.00          0.26
classic-small            124,439,808     124,439,808     0.25         36.86
modern-1b                852,559,872     852,559,872     1.71         32.77
moe-8x-top2            4,728,358,912   1,406,470,144     9.46         32.77
moe-fine-64x-top6      4,868,605,952     854,657,024     9.74         32.77

Breakdown of moe-8x-top2:
  embeddings          65,536,000    1.4%
  positions                    0    0.0%
  attention          167,772,160    3.5%
  feed_forward     4,429,185,024   93.7%
  router                 262,144    0.0%
  norms                   67,584    0.0%
  lm_head             65,536,000    1.4%
  non-embedding    4,597,286,912  (the 12*L*d^2 rule only fits dense MLPs with d_ff = 4*d)
  weights: fp32 18.91 GB, bf16 9.46 GB, int4 2.36 GB
  training compute for 1e+12 tokens ~ 6*N*D = 8.44e+21 FLOPs
  KV cache at 8,192 tokens: 0.268 GB all-global, 0.063 GB with 5 local (1024-token window) per global layer

Published mixture-of-experts sizes (sources: uv run lab facts):
  DeepSeek-V3    total 671B, active 37B per token -> 5.5% active
  Mixtral 8x7B   total 47B, active 13B per token -> 27.7% active
```

Ejecuta `uv run lab params --show classic-small` para comparar el conteo exacto con la regla de
Kaplan, y `uv run lab facts` para ver cada cifra publicada que citan las lecciones, con su fuente.

🛠️ **Constrúyelo:** reescribe `count_params` en `params.py`. `test_counter_matches_the_numpy_model_exactly`
comprueba tu contador frente a las matrices reales del modelo de numpy del paso 6.

## 💥 Rómpelo

**1. Deja de quitar los parámetros `utm_`.** En `cleaning.py`, cambia `TRACKING_PARAMS` a
`("fbclid",)` y ejecuta `uv run lab clean`. ¿Cambia la precisión? ¿Qué etapas atrapan ahora las
copias?

<details><summary>Qué deberías observar</summary>

La precisión y el *recall* no cambian (82 conservadas). `url_dedup` descarta 0 páginas; `exact_dup`
descarta ahora 11 en lugar de 8 y `quality` descarta 14 en lugar de 13, porque las copias con
seguimiento tienen el mismo texto que sus originales y una de ellas es una página de *spam*. Las etapas
en capas dan defensa en profundidad, pero cada etapa posterior hace un trabajo más caro para el mismo
resultado. Restaura el valor original.

</details>

**2. Baja el umbral de casi duplicados.** Las dos guías de calefacción independientes tienen
J = 0.417. Ejecuta el *pipeline* con un umbral de verificación de 0.3 y cuenta las páginas
conservadas:

```bash
uv run python -c "from m01_foundations.cleaning import *; print(clean_crawl(PipelineConfig(near_dup_threshold=0.3))[1].kept)"
```

¿Se fusionan las dos guías?

<details><summary>Qué deberías observar</summary>

No cambia nada: siguen conservándose 82. LSH con 16 bandas de 8 filas solo propone pares como
candidatos cuando coincide una banda entera, algo probable solo por encima de una similitud de Jaccard
de aproximadamente 0.707. Un par con 0.417 casi nunca llega a ser candidato, así que el umbral de
verificación nunca lo ve. Para cambiar lo que cuenta como casi duplicado tienes que cambiar también
las bandas.

</details>

**3. Aplica las fusiones en el orden equivocado.** En `BPETokenizer._encode_chunk`, sustituye
`min(ranked)` por `max(ranked)` y mide los *tokens* por palabra en los tickets en inglés.

<details><summary>Qué deberías observar</summary>

Los tests de ida y vuelta siguen pasando, porque decodificar solo concatena bytes. Pero los tickets en
inglés necesitan 3.360 *tokens* por palabra en lugar de 2.932: las fusiones tardías se quedan antes con
los trozos y bloquean las fusiones frecuentes que pretendía el entrenamiento. Decodificar bien no
demuestra codificar bien.

</details>

**4. Enmascara después del *softmax*.** En `scaled_dot_product_attention`, calcula primero el
*softmax* y después pon a cero los pesos enmascarados. Ejecuta `uv run pytest tests/test_transformer.py`.

<details><summary>Qué deberías observar</summary>

Fallan dos tests: `test_attention_never_looks_at_the_future` y `test_forward_is_causal_end_to_end`. Los
pesos por encima de la diagonal siguen siendo cero, pero cambiar solo el último *token* cambia las
salidas de los *tokens* anteriores: el *softmax* normalizó cada fila con las puntuaciones del futuro
antes de que las pusieras a cero. `uv run lab attention` imprime ahora `Every row sums to 1: False`. El
arreglo es enmascarar las puntuaciones con menos infinito antes del *softmax*. Restaura el código
original.

</details>

**5. Haz explotar la tasa de aprendizaje.** `uv run lab train --lr 100`.

<details><summary>Qué deberías observar</summary>

La pérdida es `nan` a partir del paso 200, y numpy avisa de valores no válidos en `softmax`. Los pasos
demasiado grandes llevan los *logits* a valores enormes hasta que la aritmética se desborda. Los
entrenamientos reales vigilan los `nan` y los picos del gradiente, y se detienen o vuelven atrás.

</details>

## 🚀 Amplíalo

**1. Arregla el descarte erróneo.** Haz que la regla de repetición se aplique solo a textos de al
menos 30 palabras y vuelve a ejecutar `uv run lab clean`.

<details><summary>Pista y solución</summary>

En `quality_failures`, protege la comprobación de repetición con el número de palabras:

```python
if s["top_trigram_share"] > cfg.max_top_trigram_share and s["words"] >= 30:
    fails.append("repetitive")
```

Todas las páginas de *spam* tienen 45 palabras o más, así que se siguen atrapando. Resultado: 83
conservadas, `precision=1.000  recall=1.000`. Recuerda el aviso de la lección 1.2: has ajustado esto
con las mismas etiquetas con las que puntúas.

</details>

**2. Mide el anonimizador.** Escribe `pii_report(docs, labels)`, que compare "el *pipeline* anonimizó
algo" con la etiqueta `has_pii` en las páginas conservadas, e imprima la precisión y el *recall*.

<details><summary>Pista y solución</summary>

Entre las páginas conservadas, 4 están etiquetadas con `has_pii`; el anonimizador toca exactamente esas
4 y ninguna más, así que la precisión y el *recall* son ambos 1.000 en este rastreo. Eso solo demuestra
que las expresiones regulares encajan con estos formatos; prueba a añadir a un test una frase como
"write to lucia at mail dot example" para ver un fallo.

</details>

**3. Elige un peso de idioma con un modelo de costes.** Alrededor del 30 % de los tickets de Hearthwise
están en español. Entrena con `--es-weight` 1, 3, 5 y 10 y calcula para cada uno un coste ponderado
por tráfico de `0.7 * en + 0.3 * es` *tokens* por palabra.

<details><summary>Pista</summary>

Reutiliza `default_tokenizer(vocab_size, es_weight)` de `cli.py` y `ticket_bodies()`. Busca el peso que
minimiza el coste ponderado y comprueba cuánto empeora el inglés. Es el mismo compromiso que asumen
quienes construyen modelos multilingües cuando fijan los pesos de muestreo de cada idioma.

</details>

**4. Añade atención de consultas agrupadas al modelo de numpy.** Dale a `TinyConfig` un campo
`n_kv_heads`, haz que `wk` y `wv` proyecten a `n_kv_heads * d_head` y repite cada cabeza de clave y
valor para su grupo de cabezas de consulta.

<details><summary>Pista</summary>

Usa `np.repeat(k, n_heads // n_kv_heads, axis=0)` después de `split_heads`. Después amplía
`test_counter_matches_the_numpy_model_exactly` con `ModelShape(..., n_kv_heads=2)` y comprueba que
`count_params` y las matrices siguen coincidiendo.

</details>

**5. Muestrea en lugar de quedarte siempre con la palabra más probable.** Añade un argumento
`temperature` a `NGramModel.generate` que muestree de la distribución de la siguiente palabra con un
`random.Random` con semilla.

<details><summary>Pista</summary>

Eleva cada probabilidad a 1/temperature, vuelve a normalizar y usa `rng.choices`. Las temperaturas
bajas se comportan como la decodificación voraz; las altas producen un texto más variado (y más
erróneo). El módulo 02 trata a fondo las estrategias de decodificación.

</details>

## ✅ Definición de terminado (*Definition of Done*)

- [ ] `uv run pytest` pasa.
- [ ] Has reconstruido al menos una función por cada paso del 2 al 8 y su archivo de tests pasa.
- [ ] Sabes explicar el único descarte erróneo de `uv run lab clean` y el *recall* de `--min-words 50`.
- [ ] Sabes explicar por qué `uv run lab bpe` da al español más *tokens* por palabra, y qué cambia `--es-weight`.
- [ ] Sabes decir cada forma que imprime `uv run lab attention` y por qué el triángulo superior es cero.
- [ ] Sabes calcular a mano los parámetros totales y activos de una configuración MoE y coincidir con `uv run lab params`.
- [ ] Has completado al menos dos experimentos de Rómpelo y una propuesta de Amplíalo.

## 🧯 Solución de problemas

| Síntoma | Solución |
|---|---|
| `uv: command not found` | Instala uv desde [docs.astral.sh/uv](https://docs.astral.sh/uv/) y abre una terminal nueva. |
| `FileNotFoundError: Could not find the repo's datasets/ folder` | Ejecuta los comandos desde dentro del repositorio clonado, o apunta `COURSE_DATASETS_DIR` a la carpeta `datasets`. |
| Números distintos de los que se muestran aquí | Asegúrate de que `datasets/` no se ha modificado: `python3 datasets/generate.py --check` desde la raíz del repositorio. |
| `RuntimeWarning: invalid value encountered` | Es lo esperado con `--lr 100`; la pérdida ha divergido a `nan`. |
| ¿De dónde ha salido `outputs/`? | `uv run lab clean` escribe ahí el corpus limpio. Git lo ignora. |

## 🔌 Usar un modelo real

Solo `uv run lab ask` habla con un proveedor de modelos; todo lo demás es autocontenido. El comando
usa la capa de proveedores del curso (`providers.py`), así que puedes apuntarlo a un asistente real con
variables de entorno:

```bash
# Local model with Ollama (https://ollama.com): start the server and pull a small model first
ollama pull llama3.2:1b
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:1b uv run lab ask "How do I reset my thermostat?"

# Any OpenAI-compatible endpoint
LLM_PROVIDER=openai OPENAI_BASE_URL=https://your-endpoint/v1 OPENAI_API_KEY=... OPENAI_MODEL=... \
  uv run lab ask "How do I reset my thermostat?"
```

La línea de estilo base no cambia (es el modelo de conteo); la línea del asistente viene del modelo
real. Espera una respuesta paso a paso que puede coincidir o no con el procedimiento real de reinicio
del Ember T3, porque el modelo nunca ha visto el centro de ayuda de Hearthwise. Esa carencia es la que
cierran los módulos 05 y 06.

Estado sincero: estos caminos en vivo **no se ejecutaron** cuando se escribió este lab (no había
ningún servidor de Ollama en marcha ni se usó ninguna clave compatible con OpenAI). Los tests solo
comprueban, sin conexión, que las peticiones a los proveedores se construyen correctamente; los tests
en vivo se marcarían con `ollama` u `openai` y solo se ejecutarían con `RUN_LIVE_TESTS=1`. Si algo
falla, `uv run pytest
tests/test_providers.py` y el mensaje de error de `ProviderError` son los sitios por donde empezar.

---

[← Anterior: 1.5 Familias de modelos](../05-model-families.es.md) · [🏠 Módulo 01](../README.es.md) · [Siguiente: Quiz →](../quiz.es.md)
