🌐 [English](README.md) · Español

# Lab 02 — Decodificación, datos de post-entrenamiento y evaluación desde cero

> Construye todas las estrategias de decodificación sobre un modelo de lenguaje de Hearthwise, limpia un
> dataset de SFT, entrena un modelo de recompensa que cae en la trampa de la longitud y evalúa modelos
> con métricas, un detector de contaminación, una clasificación y una auditoría del juez. ⏱️ unas 3 horas.

## 🎯 Qué vas a construir

| Archivo | Qué construyes | Lección |
|---|---|---|
| `src/m02_post_training/ngram.py` | Un modelo de lenguaje de trigramas que devuelve una log-probabilidad por entrada del vocabulario | 2.1, 2.4 |
| `src/m02_post_training/decoding.py` | *Softmax* con temperatura, decodificación voraz, búsqueda en haz, *top-k*, *top-p*, penalizaciones, secuencias de parada, semillas, métricas de diversidad | 2.1 |
| `src/m02_post_training/sft.py` | Una plantilla de chat, una máscara de pérdida y un limpiador de datos puntuado frente a etiquetas | 2.2 |
| `src/m02_post_training/reward.py` | Una recompensa verificable, un modelo de recompensa de Bradley–Terry, un test de explotación, la pérdida de DPO y la recompensa penalizada con KL | 2.3 |
| `src/m02_post_training/evaluation.py` | EM, F1, BLEU, ROUGE-L, perplejidad, un detector de contaminación, puntuaciones Elo y Bradley–Terry, un test de intercambio para el juez | 2.4 |
| `src/m02_post_training/cli.py` | El punto de entrada `uv run lab <command>` | todas |

`providers.py` y `data.py` son los archivos compartidos del curso, copiados sin cambios. Todo se ejecuta
sin conexión en una CPU con una semilla fija, así que tus números deben coincidir exactamente con los de
abajo.

**Cómo trabajarlo.** El paquete contiene una implementación de referencia completa, así que los tests
pasan desde el principio. Cada paso nombra las funciones que debes reconstruir. Para cada una: lee los
tests que la cubren, borra el cuerpo de la función (conserva la firma y el *docstring*), escribe tu
propia versión y ejecuta los tests y el comando del paso hasta que la salida vuelva a coincidir. Si te
atascas, `git diff` muestra lo que cambiaste y `git checkout -- <file>` recupera la referencia.

## ⚙️ Preparación

```bash
cd modules/02-decoding-post-training-evaluation/lab
uv sync
uv run pytest
```

Resumen esperado (el tiempo es un ejemplo y varía según la máquina):

```text
........................................................................ [ 80%]
.................                                                        [100%]
89 passed in 8.09s
```

`uv run lab --help` lista todos los comandos.

## 👣 Pasos

### Paso 1 — Entrena el modelo base y mira sus logits

Reconstruye `NgramLM.next_logprobs` en `ngram.py` y `softmax` en `decoding.py`. El modelo mezcla
estimaciones de trigramas, bigramas y frecuencia de palabras con pesos 0,6, 0,3 y 0,1; cuando un
contexto no se vio nunca, su peso pasa a los órdenes inferiores.

```bash
uv run lab corpus
uv run lab next-token --prompt "to reset the"
```

```text
documents: 95 (17 help-centre pages, 78 web pages)
training tokens (including <eos>): 12544
vocabulary (words seen at least twice + 3 special tokens): 797
prompt: 'to reset the'
T=0.5  'device' 0.996  'router' 0.000  'app' 0.000  'thermostat' 0.000  'home' 0.000
T=1.0  'device' 0.612  'router' 0.011  'app' 0.009  'thermostat' 0.009  'home' 0.008
T=2.0  'device' 0.063  'router' 0.009  'app' 0.008  'thermostat' 0.007  'home' 0.007
tokens kept by top-p (p=0.5: 1, p=0.9: 79) out of 797
```

### Paso 2 — Compara las estrategias de decodificación

Reconstruye `top_k_filter`, `top_p_filter`, `apply_repetition_penalty`, `beam_search`, `distinct_n` y
`repetition_rate` en `decoding.py`. Después provoca un bucle voraz, rómpelo y ejecuta el *benchmark*.

```bash
uv run lab generate --strategy greedy --prompt "returns are" --max-tokens 50
uv run lab generate --strategy greedy --prompt "returns are" --max-tokens 50 --repetition-penalty 1.3
uv run lab generate --top-p 0.9 --seed 7
uv run lab decode-bench
```

```text
returns are -> hard to debug. avoid bright cold light late in the app with the appliance, not the lamp, so voice commands stay simple. warm white works best in the app with the appliance, not the lamp, so voice commands stay simple. warm white works
[finish_reason=length tokens=50 logprob=-29.91 rep-3=0.38]
returns are -> hard to debug. avoid bright cold light late in the app with the appliance, not the lamp, so voice commands stay simple. warm white works best in the bedroom and see if anyone notices. a smart plug can switch on when the last person leaves
[finish_reason=length tokens=50 logprob=-34.15 rep-3=0.00]
to reset the -> device to support 5 ghz wi-fi) battery sensors last longer than wi-fi ones. name your gadgets a week before adding more; future you will thank you. how do? tap forgot password on. open the
[finish_reason=length tokens=40 logprob=-48.08 rep-3=0.00]
6 prompts x 5 seeds, up to 50 new tokens each
strategy               distinct-2  rep-3  unique  logp/tok  ended
greedy                      0.130  0.062    6/30     -0.78   0.83
beam (4 beams)              0.140  0.000    6/30     -0.85   1.00
sample T=1.0                0.540  0.021   30/30     -2.00   0.03
sample T=0.7                0.532  0.006   30/30     -1.12   0.40
sample T=1.5                0.623  0.000   30/30     -5.77   0.40
top-k k=10                  0.616  0.008   30/30     -1.26   0.27
top-p p=0.9                 0.635  0.001   30/30     -1.38   0.37
top-p 0.9 + rep 1.3         0.655  0.003   30/30     -1.32   0.23
```

Ejecuta dos veces el comando con la semilla 7: el texto debe ser idéntico. Después prueba `--seed 8`.

### Paso 3 — Renderiza plantillas de chat y máscaras de pérdida

Reconstruye `render_chat` y `loss_mask` en `sft.py`.

```bash
uv run lab chat-template --id s002
```

```text
<|system|>
You are Hearthwise's support assistant. Answer briefly, accurately and kindly.<|end|>
<|user|>
How many devices can the Hearth Hub handle? Thanks.<|end|>
<|assistant|>
The Hearth Hub connects up to 64 HomeMesh devices.<|end|>
--- loss mask ([x] = trained) ---
<|system|> You are Hearthwise ' s support assistant . Answer briefly , accurately and kindly . <|end|> <|user|> How many devices can the Hearth Hub handle ? Thanks . <|end|> <|assistant|> [The] [Hearth] [Hub] [connects] [up] [to] [64] [HomeMesh] [devices] [.] [<|end|>]
tokens: 42  trained: 11  (26%)
```

### Paso 4 — Limpia los datos de SFT

Reconstruye `is_low_quality`, `has_pii`, `ContaminationIndex.matches` y `clean` en `sft.py`.

```bash
uv run lab sft-clean
```

```text
rows: 261  kept: 108  dropped: 153
check                    labelled  flagged  correct precision  recall
contaminated_rag_eval           5       11        5      0.45    1.00
empty_response                  1        1        1      1.00    1.00
low_quality                     1        1        1      1.00    1.00
outdated_policy                 4        4        4      1.00    1.00
pii_in_response                 3        3        3      1.00    1.00
wrong_language                  1        1        1      1.00    1.00
duplicate                dropped 132 exact repeats; labelled duplicates kept exactly once: 6/6
kept rows that still ask an eval question word for word: 5
distinct responses among kept rows: 25
tokens before cleaning: 13874 (42% trained)  after: 5969 (42% trained)
```

Después abre las 6 filas que la comprobación de contaminación señala pero las etiquetas no, y decide
quién tiene razón.

### Paso 5 — Entrena un modelo de recompensa y detecta la trampa de la longitud

Reconstruye `fit_reward_model`, `pairwise_accuracy`, `prompt_disjoint` y `hack_rate` en `reward.py`.

```bash
uv run lab reward
```

```text
'longer answer wins' baseline: train 0.787  test 0.233
padding used for the hack test: 'Thank you so much for reaching out to Hearthwise, we really appreciate you taking the time to contact us.'

official split (same prompts in train and test)
reward model               pairs feats  train   test   hack
length only                  240     1  0.787  0.233  0.767
words + length               240   302  1.000  1.000  0.000

new prompts (train and test questions do not overlap)
reward model               pairs feats  train   test   hack
length only                  125     1  0.784  0.250  0.750
words + length               125   217  1.000  0.714  0.357
words + length, balanced      54   217  1.000  0.821  0.250

words + length (new prompts), largest weights:
  + or 0.76, <length> 0.76, 30 0.48, orders 0.47, no 0.47
  - yes -0.62, 5 -0.38, works -0.38, 14 -0.37, 1-year -0.31
```

### Paso 6 — Calcula las magnitudes de DPO y PPO

Reconstruye `verifiable_reward`, `dpo_loss` y `kl_penalized_reward` en `reward.py`.

```bash
uv run lab verify
uv run lab dpo
```

```text
r001: A customer buys 3 x Spark Mini at €19 each, 1 x Ember T3 at €149 each. A 15% discount code applies to the products (not to shipping). Shipping is free when the products cost more than €49 after any discount; otherwise it costs €4.95. What is the total in euros?
gold answer: 175.10
reward 1.0 <- '206 x 0.85 = 175.10, shipping is free, so the total is 175.10 euros.'
reward 1.0 <- 'The total is €175.1'
reward 0.0 <- 'Subtotal 206, minus 15% is 175.10, plus €4.95 shipping = 180.05'
reference model: log p(chosen) = -20.0, log p(rejected) = -18.0
policy scenario                            margin loss b=0.1 loss b=0.5
policy = reference (start of training)        0.0      0.693      0.693
chosen up 1, rejected down 1                  2.0      0.598      0.313
chosen down 2, rejected down 5                3.0      0.554      0.201
chosen down 1, rejected up 1 (wrong way)     -2.0      0.798      1.313

PPO-style reward: r=2.0, log pi=-12.0, log pi_ref=-15.0, beta=0.1 -> 1.70
```

### Paso 7 — Puntúa respuestas con métricas tradicionales

Reconstruye `token_f1`, `bleu` y `rouge_l` en `evaluation.py` y `NgramLM.perplexity` en `ngram.py`.

```bash
uv run lab metrics
```

```text
reference answer (rag_eval q001): '30 days from delivery.'
candidate                  correct    EM    F1  BLEU ROUGE-L
copy of the reference          yes  1.00  1.00  1.00    1.00
correct paraphrase             yes  0.00  0.46  0.18    0.53
correct, different words       yes  0.00  0.36  0.14    0.40
wrong (old policy)              no  0.00  0.75  0.75    0.80
correct but verbose            yes  0.00  0.33  0.14    0.34

perplexity of a trigram model trained on 90% of the corpus
text                       tokens  <unk> perplexity
training documents          11043     4%       2.54
held-out documents           1406     6%       7.01
greedy outputs                219     0%       2.66
top-p 0.9 outputs             128     0%       9.80
T=1.5 outputs                  97     0%     674.91
Spanish help-centre pages     266    80%      20.30
```

### Paso 8 — Detecta la contaminación de benchmarks

Reconstruye `item_in_page` y `find_contamination` en `evaluation.py`.

```bash
uv run lab contamination
```

```text
web pages labelled as contaminated: 2; eval questions copied word for word: 12
  n share pages flagged correct items flagged correct
  3   0.5             2       2            46      12
  8   0.5             2       2            45      12
  8   1.0             2       2            12      12
 13   1.0             2       2            12      12
https://studyhub.example/math-practice-with-solutions: r001, r002, r003, r004, r005, r006
https://studyhub.example/hearthwise-support-quiz-answers: q001, q002, q003, q004, q005, q006
```

### Paso 9 — Construye una clasificación y audita un juez

Reconstruye `elo_ratings`, `bradley_terry_ratings` y `judge_swap_test` en `evaluation.py`. Los votos se
simulan a partir de fuerzas ocultas para que puedas comprobar que un método las recupera; el juez es un
sustituto guionizado con una preferencia por la longitud incorporada y una bonificación de cinco
palabras para la respuesta que se muestra primero.

```bash
uv run lab arena
uv run lab judge
```

```text
600 simulated votes between 4 models
model     truth    Elo Elo rev     BT  BT 95% interval
model-A    1113   1105    1087   1105     1075 to 1146
model-B    1026   1021    1078   1026      997 to 1061
model-C     991    986     985    993      959 to 1022
model-D     870    887     850    875      844 to  903
Elo ranking:          model-A > model-B > model-C > model-D
Elo ranking reversed: model-A > model-B > model-C > model-D
BT ranking:           model-A > model-B > model-C > model-D
judge provider: fake
pairs                          n agree AB agree BA consistent  first longer
test pairs as rated           60    0.233    0.233      1.000  0.500  1.000
test pairs, filler removed    60    1.000    0.667      0.667  0.667  0.833
```

## 💥 Rómpelo

1. **Quita el contexto al modelo.** En `cli.build_lm`, crea `NgramLM(lambdas=(1.0, 0.0, 0.0))` y
   ejecuta `uv run lab generate --strategy greedy --max-tokens 12`. ¿Qué dice un "modelo de lenguaje"
   que solo conoce frecuencias de palabras?

   <details><summary>Qué deberías ver</summary>

   Una fila de puntos. Con solo frecuencias de palabras, el *token* más común es siempre el siguiente
   más probable, así que la decodificación voraz lo elige para siempre. El contexto es lo que convierte
   las frecuencias en lenguaje. Después, vuelve a poner las lambdas por defecto.

   </details>

2. **Un error por uno en *top-p*.** En `top_p_filter`, quita el `+ 1` de `n_keep` y ejecuta
   `uv run lab generate --top-p 0.5 --seed 1`.

   <details><summary>Qué deberías ver</summary>

   Un fallo: `ValueError: Probabilities contain NaN`. Tras "to reset the", "device" ya supera 0,5 por sí
   solo, así que la posición de búsqueda es 0 y el filtro conserva cero *tokens*. Cada *logit* pasa a
   menos infinito y *softmax* divide cero entre cero. Los tests `test_top_p_keeps_smallest_set_reaching_p`
   también lo detectan.

   </details>

3. **Olvida entrenar el marcador de fin.** En `loss_mask`, añade 0 en lugar de 1 para `END` dentro de
   un turno del asistente y ejecuta `uv run pytest tests/test_sft.py`.

   <details><summary>Qué deberías ver</summary>

   Falla `test_loss_mask_trains_only_assistant_tokens_and_end_marker`. Un modelo entrenado con esta
   máscara nunca practica emitir `<|end|>`, así que no sabría cuándo parar: la divagación de la lección
   2.1.

   </details>

4. **Oculta el rasgo de longitud.** En `reward._run`, construye el extractor de rasgos con
   `use_length=False` y ejecuta `uv run lab reward`. ¿Baja a cero la tasa de explotación con *prompts*
   nuevos?

   <details><summary>Qué deberías ver</summary>

   No. La exactitud en prueba y la tasa de explotación apenas se mueven, porque las palabras de relleno
   ("thank", "appreciate", "wonderful") aparecen casi solo en las respuestas largas y arrastran el mismo
   sesgo. Quitar el rasgo explícito de longitud no elimina el sesgo de longitud; equilibrar los datos
   consigue más.

   </details>

5. **Quita la bonificación de posición del juez.** En `cli.cmd_judge`, pasa `position_bonus=0` a
   `biased_judge_responder` y vuelve a ejecutar `uv run lab judge`.

   <details><summary>Qué deberías ver</summary>

   En los pares sin relleno, la coherencia sube mucho y la tasa de "elegida la primera" se acerca a 0,5.
   Queda un pequeño efecto de posición, porque los empates de longitud siguen yendo a la respuesta A
   (`>=`). Es el test de intercambio haciendo su trabajo: mide exactamente el sesgo que plantaste.

   </details>

## 🚀 Amplíalo

1. **Añade un filtro min-p.** Conserva solo los *tokens* cuya probabilidad sea al menos `m` veces la
   probabilidad del *token* más probable. Añádelo a `DecodeConfig` y a `generate`, añade una estrategia a
   `BENCH_STRATEGIES` y compáralo con *top-p* en `decode-bench`.

   <details><summary>Pista y solución</summary>

   ```python
   def min_p_filter(logits: np.ndarray, m: float, temperature: float = 1.0) -> np.ndarray:
       """Keep tokens with probability >= m * (largest probability)."""
       if m <= 0:
           return logits
       probs = softmax(logits, temperature)
       return np.where(probs >= m * probs.max(), logits, NEG_INF)
   ```

   A diferencia de *top-p*, min-p ignora cuánto suma la cola, así que una cola larga y plana se corta
   aunque acumule mucha masa en total.

   </details>

2. **Añade una penalización por longitud a la búsqueda en haz.** Elige el haz final por
   `score / len(seq) ** alpha` en lugar de `score`, y mira cómo cambia la respuesta en haz a "to reset
   the" para `alpha` = 0,5 y 1,0.

   <details><summary>Pista</summary>

   Solo hace falta cambiar el `max(pool, ...)` final, más un nuevo campo `length_penalty` en
   `DecodeConfig`. Protégete de las secuencias vacías (`max(len(seq), 1)`).

   </details>

3. **Deduplica solo por la respuesta.** Añade una opción `dedup_by_response` a `sft.clean` y cuenta
   cuántas filas sobreviven. ¿Un saludo parafraseado ("Hi! ...", "... Thanks.") es un ejemplo nuevo o un
   duplicado?

   <details><summary>Pista</summary>

   Usa `" ".join(tokenize(row["response"]))` como clave. La respuesta a la pregunta es una cuestión de
   criterio: las variantes del saludo enseñan robustez ante la redacción, pero repiten el mismo texto
   objetivo. Escribe tu decisión en un comentario.

   </details>

4. **¿Cuántos votos bastan?** Llama a `evaluation.simulate_votes(n_votes=...)` con cantidades mayores y
   encuentra la cantidad más pequeña con la que los intervalos *bootstrap* de model-B y model-C dejan de
   solaparse.

   <details><summary>Pista</summary>

   Prueba a duplicar: 600, 1.200, 2.400. Con 2.400 votos los dos intervalos ya no se solapan. Este es el
   cálculo que debes hacer antes de pedir tiempo a los agentes de Alex.

   </details>

5. **Cuenta solo los veredictos coherentes.** Añade a `JudgeReport` un campo con el acuerdo calculado
   solo sobre los pares en los que ambos órdenes dieron el mismo ganador, y la proporción de pares
   descartados.

   <details><summary>Pista</summary>

   Mantén dos contadores dentro del bucle: uno para los pares coherentes y otro para los pares
   coherentes cuyo ganador es la respuesta elegida.

   </details>

## ✅ Definición de terminado (*Definition of Done*)

- [ ] `uv run pytest` pasa con tus propias implementaciones de las funciones nombradas en los pasos 1 a 9.
- [ ] Cada comando de los pasos imprime exactamente la salida esperada de arriba.
- [ ] Sabes explicar el bucle voraz y cómo lo rompe la penalización por repetición.
- [ ] Revisaste los 6 aciertos de contaminación sin etiqueta en los datos de SFT y sabes decir por qué son fugas.
- [ ] Sabes explicar por qué la exactitud en prueba de 1,000 del modelo de recompensa en la partición oficial es engañosa.
- [ ] Completaste al menos dos experimentos de "Rómpelo" y una tarea de "Amplíalo".
- [ ] Apuntaste qué evaluación usarías para la elección de modelo de Hearthwise y por qué.

## 🧯 Solución de problemas

| Problema | Solución |
|---|---|
| `uv: command not found` | Instala uv desde la [documentación oficial](https://docs.astral.sh/uv/). |
| `FileNotFoundError: Could not find the repo's datasets/ folder` | Ejecuta los comandos desde dentro del repositorio, o define `COURSE_DATASETS_DIR` con la carpeta `datasets`. |
| Los números difieren ligeramente de esta página | Comprueba que no cambiaste ningún valor por defecto (semillas, lambdas, umbrales) y que `uv sync` instaló las versiones fijadas. Restaura un archivo con `git checkout -- <file>` para comparar. |
| `ValueError: Probabilities contain NaN` | Un filtro eliminó todos los *tokens*; consulta el punto 2 de "Rómpelo". |
| `RuntimeError: Network access is disabled in tests` | Un test intentó conectarse a un servidor. Solo pueden hacerlo los tests marcados con `ollama` u `openai`, y solo con `RUN_LIVE_TESTS=1`. |

## 🔌 Usar un modelo real

Casi todo este lab trata de algoritmos que solo se pueden ver dentro de un modelo pequeño, así que no
necesita un modelo real. El único sitio donde tiene sentido un modelo real es el **juez** del paso 9:
`lab judge` envía sus *prompts* a través de `providers.py`, así que puede llamar a un modelo local o
alojado en lugar del sustituto guionizado.

```bash
# Local model with Ollama (https://ollama.com)
ollama pull llama3.2:1b
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:1b uv run lab judge

# Any OpenAI-compatible endpoint
LLM_PROVIDER=openai OPENAI_BASE_URL=https://your-endpoint/v1 OPENAI_API_KEY=... OPENAI_MODEL=... uv run lab judge
```

El juez envía 240 peticiones (60 pares, dos órdenes, dos conjuntos de pares), así que un *endpoint*
alojado costará dinero. Un modelo local pequeño puede no responder con una sola letra; entonces
`parse_verdict` devuelve `None`, que el informe cuenta como un voto para la respuesta B. Mira algunas
respuestas en bruto antes de fiarte de los números.

También puedes enviar a un modelo real los ajustes de decodificación de la lección 2.1. `providers.py`
pasa la temperatura, el máximo de *tokens*, la semilla y las secuencias de parada; las opciones de Ollama
también aceptan `top_k`, `top_p` y `repeat_penalty`, que puedes añadir a
`OllamaProvider.build_chat_payload` como ampliación.

**Sin probar:** las rutas de Ollama y de *endpoints* compatibles con OpenAI no se ejecutaron cuando se
escribió este lab. En CI solo se ejecutan los tests de contrato de proveedores sin conexión de
`tests/test_providers.py`; ningún test de este lab llama a un modelo en vivo.

---

[← Anterior: 2.4 Evaluar LLM](../04-evaluating-llms.es.md) · [🏠 Módulo 02](../README.es.md) · [Siguiente: Quiz →](../quiz.es.md)
