🌐 [English](README.md) · Español

# Lab 09 — Razonamiento al responder y al entrenar, medido sin conexión

> Ejecuta respuestas directas, con cadena de pensamiento y con pensamiento, y después construye y mide
> la autoconsistencia, los verificadores, el árbol de pensamientos, la búsqueda en haz con PRM, un
> bucle STaR y un juguete de RL al estilo de GRPO frente a un razonador simulado con semilla. ⏱️ unas 3
> horas.

## 🎯 Qué vas a construir

Todos los comandos hablan con la misma interfaz de chat (`FakeProvider`, de la capa de proveedores del
curso). Por defecto, el "modelo" que hay detrás es un **razonador simulado**: un programa con semilla
escrito para este lab, no un modelo de lenguaje. Escribe soluciones paso a paso a los 160 problemas de
`datasets/reasoning/problems.jsonl` con el mismo formato que sus pasos de referencia, y comete a
propósito tres tipos de errores: deslices (un número erróneo en un paso), trampas (una lectura errónea
y coherente de la pregunta, como contar las bombillas Wi-Fi como dispositivos del hub) y aciertos por
suerte (respuesta correcta, paso erróneo). Eso hace que cada algoritmo sea observable y cada número,
reproducible en la CPU de un portátil. Los números muestran cómo se comportan los algoritmos; no son
puntuaciones de ningún modelo real.

| Archivo en `src/m09_reasoning/` | Qué hace | Lección |
|---|---|---|
| `problems.py` | Carga los problemas; comprobaciones exactas de las respuestas; el verificador de expresiones para los acertijos de número objetivo | 9.1 |
| `prompts.py` | Los *prompts* de sistema y los formatos de mensaje (directo, CoT, pensamiento, siguiente paso, revisión, propuesta, valor) | 9.1 |
| `simulator.py` | El razonador simulado: los mandos de `SimConfig`, deslices, trampas, autocomprobaciones | 9.1 |
| `reasoner.py` | El cliente: construye los *prompts*, analiza los pasos y las respuestas, cuenta los *tokens* (`Usage`) | 9.1 |
| `scaling.py` | Votación por mayoría, *best-of-N*, muestreo, bucles de revisión y fuentes de comentarios | 9.2 |
| `verifiers.py` | Recomprobación aritmética (exacta); ORM y PRM simulados | 9.3 |
| `search.py` | Búsqueda en haz con PRM, árbol de pensamientos (BFS y DFS), muestreo hasta verificar | 9.3 |
| `training.py` | STaR (con racionalización y un filtro estricto); ventajas de grupo; juguete al estilo de GRPO | 9.4 |
| `experiments.py` | Cada experimento que hay detrás de los números de las lecciones | todas |
| `exercises.py` | Tres funciones para que las implementes | 9.1, 9.2 |
| `cli.py` | `uv run lab <command>` | todas |

## ⚙️ Preparación

Desde la raíz del repositorio:

```bash
cd modules/09-reasoning-models/lab
uv sync
uv run pytest
```

Resumen esperado (el tiempo varía según la máquina):

```text
54 passed, 3 skipped
```

Los 3 tests omitidos comprueban los ejercicios del paso 10; se ejecutan en cuanto los implementes. No
hace falta clave de API, red ni GPU. `uv run lab --help` lista los comandos: `solve`, `scale`,
`revise`, `search`, `tot`, `star` y `rl`.

## 👣 Pasos

### Paso 1 — Prepara el entorno y ejecuta los tests

Haz la preparación de arriba. Después abre `tests/test_simulator.py` y lee
`test_without_errors_the_simulator_reproduces_every_reference_solution`. Demuestra que, cuando el
simulador no comete ningún error, escribe exactamente los pasos de referencia del dataset para los 160
problemas. Cada error que verás más adelante se inyectó a propósito y quedó registrado.

### Paso 2 — Conoce al razonador simulado

```bash
uv run lab solve --id r076
```

```text
r076 · hub_capacity · difficulty 5
A Hearth Hub can connect up to 64 HomeMesh devices. ...
Reference answer: 52

=== direct: 3 tokens (0 thinking), wrong
Answer: 12

=== cot: 85 tokens (0 thinking), wrong
Step 1: HomeMesh devices per room: 1 motion + 1 door + 5 bulbs = 7
Step 2: Rooms: 2 floors x 3 rooms = 6
Step 3: HomeMesh devices: 6 x 7 = 52
Next I need the following quantity.
Step 4: Cameras use Wi-Fi, so they do not count
I will keep two decimals for euro amounts.
Step 5: Free slots: 64 - 52 = 12
Answer: 12
[behind the curtain: step kinds trap, ok, slip, ok, ok]
...
=== think-medium: 329 tokens (254 thinking), correct
```

La línea `[behind the curtain]` es el registro propio del simulador: el paso 1 cayó en la trampa (las
bombillas) y el paso 3 tuvo un desliz. Lee las salidas de think-low y think-medium y averigua qué error
detecta cada una. Después abre `simulator.py` y lee `SimConfig.slip_probability` y
`SimConfig.trap_probability`: todo el modelo de errores está en esas dos funciones y en los métodos
`write` de las cuatro familias de problemas.

### Paso 3 — Compara las respuestas directas, CoT y con pensamiento

```bash
uv run lab solve
```

```text
mode          accuracy  diff 2  diff 3  diff 4  diff 5  tokens/answer  of which thinking
------------  --------  ------  ------  ------  ------  -------------  -----------------
direct           33.1%   44.7%   28.1%   40.0%   22.9%            6.5                0.0
cot              49.4%   57.9%   38.6%   63.3%   45.7%           72.7                0.0
think-low        57.5%   63.2%   49.1%   70.0%   54.3%          138.3               80.5
think-medium     68.1%   65.8%   54.4%   86.7%   77.1%          161.2              103.5
think-high       71.9%   71.1%   56.1%   86.7%   85.7%          194.1              136.4
```

Tu turno: en `reasoner.py`, lee `parse_solution` y busca dónde se cuentan los *tokens* de
pensamiento. Después responde para Sam: ¿qué columna pondrías en un panel de costes, y por qué no la
longitud de la respuesta visible?

### Paso 4 — Escala con muestras en paralelo

```bash
uv run lab scale
```

```text
N   majority vote  best-of-N (ORM)  pass@N (oracle)  tokens/problem
--  -------------  ---------------  ---------------  --------------
1           61.8%            61.8%            61.8%            77.3
3           69.1%            82.7%            87.3%           230.9
5           76.4%            86.4%            92.7%           386.2
9           85.5%            86.4%            97.3%           695.9
15          89.1%            91.8%           100.0%         1,165.0
45          89.1%            93.6%           100.0%         3,515.6

At N=45: the most frequent answer is wrong on 12 of 110 problems; 12 of them are trap answers. Problems never solved: 0.
```

Lee `majority_vote`, `answer_key` y `best_of_n` en `scaling.py`, y `vote_stats` (que explica el
estancamiento). Tu turno: implementa `weighted_vote` y `adaptive_vote` en `exercises.py` (paso 10).

### Paso 5 — Revisa con comentarios

```bash
uv run lab revise
```

```text
feedback              round 0  round 1  round 2  round 3  tokens/problem  calls/problem  right->wrong
--------------------  -------  -------  -------  -------  --------------  -------------  ------------
self-critique           61.8%    65.5%    69.1%    66.4%           155.9           4.92             9
arithmetic re-check     61.8%    81.8%    85.5%    85.5%           101.6           1.38             0
process reward model    61.8%    80.9%    90.0%    86.4%           152.8           2.27             6
```

Lee `revise_loop` y las tres funciones de comentarios en `scaling.py`. Fíjate en que cada fuente de
comentarios devuelve o bien `None` (parar) o bien una frase que nombra un paso; al bucle le da igual de
dónde venga.

### Paso 6 — Busca con verificadores

```bash
uv run lab search --id r076
uv run lab search
```

```text
depth 1: 2 distinct candidates, keep 2
  keep 0.95  HomeMesh devices per room: 1 motion + 1 door = 2
  keep 0.46  HomeMesh devices per room: 1 motion + 1 door + 5 bulbs = 7
...
Answer 52 (correct), 39 calls, 468 generated tokens, 6391 prompt tokens
```

```text
method                         accuracy  generated tok  prompt tok  calls
-----------------------------  --------  -------------  ----------  -----
CoT, 1 sample                     61.8%           77.3       110.4    1.0
majority vote, N=9                85.5%          695.9       993.9    9.0
re-check filter + vote, N=9       86.4%          695.9       993.9    9.0
best-of-9, ORM                    86.4%          695.9       993.9    9.0
best-of-9, PRM                    95.5%          695.9       993.9    9.0
PRM beam search, 3x3              93.6%          274.3     3,055.8   20.5
pass@9 (needs the answer key)     97.3%          695.9       993.9    9.0

Verifier errors over all 4,950 samples (1,854 wrong, 801 of them traps):
verifier               wrong it lets through  right it rejects
---------------------  ---------------------  ----------------
arithmetic re-check                      572               238
ORM (score >= 0.5)                       571               518
PRM (min step >= 0.5)                     64               906

Reward hacking: best-of-N with an honest ORM vs one that also likes long answers (length bias 5.0)
N   honest ORM  length-biased ORM  tokens of pick (honest)  tokens of pick (biased)
--  ----------  -----------------  -----------------------  -----------------------
1        61.8%              61.8%                     77.3                     77.3
3        82.7%              73.6%                     74.1                     79.5
5        86.4%              74.5%                     72.6                     81.4
9        86.4%              77.3%                     73.8                     82.6
15       91.8%              79.1%                     76.4                     86.1
45       93.6%              80.9%                     77.5                     88.4
```

Lee `step_is_consistent` en `verifiers.py` (la recomprobación exacta), `SimulatedORM.score` y
`SimulatedPRM.step_scores`, y `prm_beam_search` en `search.py`.

### Paso 7 — Árbol de pensamientos en los acertijos

```bash
uv run lab tot --id r113
uv run lab tot
```

```text
=== DFS
try [2 4 1] (rated likely)
  prune [2 3] (rated impossible)
...
try [2 12 11/4] (rated likely)
  try [11/4 24] (rated sure)
    no last move reaches the target: backtrack
...
try [2 11 3] (rated likely)
  try [11 1] (rated sure)
found: 11 + ((12 / 4) - 2)  ·  26 calls, 201 generated tokens
```

```text
method                         solved  generated tok  prompt tok  calls  verifier rejections
-----------------------------  ------  -------------  ----------  -----  -------------------
CoT, 1 sample                   11/50           62.1        78.4    1.0                   39
sample until verified, max 5    30/50          218.6       263.7    3.4                  138
sample until verified, max 15   41/50          406.6       485.7    6.2                  268
ToT BFS, breadth 1              27/50           84.6       922.7   11.1                    0
ToT BFS, breadth 3              35/50          152.9     1,657.5   19.9                    0
ToT BFS, breadth 5              36/50          218.6     2,381.9   28.6                    0
ToT DFS with pruning            34/50           87.9     1,032.0   12.4                    0
```

Lee `tot_bfs` y `tot_dfs` en `search.py`, y después `verify_expression` en `problems.py`. "Verifier
rejections" cuenta las respuestas que rechazó el verificador exacto: cada muestra fallida en las filas
de muestreo, y las falsas afirmaciones de "he llegado al objetivo" en las filas del árbol.

### Paso 8 — Entrena con STaR

```bash
uv run lab star
uv run lab star --no-rationalize
uv run lab star --strict
```

```text
round  test accuracy  energy  hub_capacity  order_total  target  kept (solved)  kept (hint)  unfaithful kept  tokens spent
-----  -------------  ------  ------------  -----------  ------  -------------  -----------  ---------------  ------------
0              51.2%   82.4%         38.9%        75.0%   20.0%             40           27                5         8,625
1              68.8%  100.0%         55.6%        80.0%   48.0%             51           21                1         7,508
2              71.2%  100.0%         61.1%        80.0%   52.0%             58           19                3         6,770
3              73.8%  100.0%         66.7%        85.0%   52.0%             56           21                2         7,178
4              72.5%  100.0%         61.1%        85.0%   52.0%              -            -                -             -
```

La última fila de las otras dos ejecuciones: sin racionalización, el bucle termina en 62.5% (acertijos
de número objetivo, 28.0%); con el filtro estricto termina en 73.8% y no conserva ninguna
justificación infiel. Lee `collect_rationales` y `finetune` en `training.py`. Solo `finetune` es
simulado; todo lo demás es el bucle real.

### Paso 9 — Refuerza con una recompensa verificable

```bash
uv run lab rl
```

```text
step  train reward  test accuracy  tokens/answer  P(check step)  P(re-read)  groups with no signal
----  ------------  -------------  -------------  -------------  ----------  ---------------------
0            0.514          53.8%           77.9           0.10        0.10                     10
1            0.530          55.0%           79.8           0.11        0.19                     17
2            0.503          55.0%           81.2           0.09        0.38                     17
3            0.534          60.0%           90.4           0.38        0.70                     18
4            0.606          66.2%          108.8           0.95        0.77                     18
5            0.670          67.5%          112.5           0.98        0.93                     25
6            0.667          67.5%          112.6           0.99        0.93                     18
7            0.658          68.8%          113.0           0.99        0.94                     20
8                -          68.8%          113.0           0.99        0.94                      -
```

Lee `group_advantages` y `grpo_toy` en `training.py`. La actualización es REINFORCE con ventajas
relativas al grupo; el GRPO real añade el recorte (*clipping*) y una penalización por alejarse de un
modelo de referencia. Los dos parámetros son los *logits* de "comprobar cada paso" y "releer la
pregunta".

### Paso 10 — Tu turno: tres ejercicios

Abre `src/m09_reasoning/exercises.py` e implementa las tres funciones. `uv run pytest
tests/test_exercises.py` ejecuta sus tests (se omiten hasta que cada función deja de lanzar
`NotImplementedError`).

1. `pass_at_k(n, c, k)`: el estimador insesgado de pass@k (lección 9.1).
2. `weighted_vote(answers, scores)`: una votación en la que cada muestra cuenta con su puntuación del
   verificador.
3. `adaptive_vote(draw, max_n, lead)`: autoconsistencia que se detiene en cuanto la respuesta que va
   en cabeza está a salvo.

Después úsalas: compara `adaptive_vote` con `lead=3, max_n=15` frente a la votación fija con N = 15 en
los 110 problemas numéricos, contando las muestras usadas. Usa `make_reasoner()` y
`lambda: reasoner.solve(p).answer` como función `draw`.

<details><summary>Solución: pass_at_k</summary>

```python
from math import comb

def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:  # every set of k samples contains a correct one
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
```

</details>

<details><summary>Solución: weighted_vote</summary>

```python
from collections import defaultdict
from m09_reasoning.scaling import answer_key

def weighted_vote(answers, scores):
    totals: dict[str, float] = defaultdict(float)
    first: dict[str, str] = {}
    for answer, score in zip(answers, scores):
        key = answer_key(answer)
        if key is None or answer is None:
            continue
        totals[key] += score
        first.setdefault(key, answer)
    if not totals:
        return None
    best = max(totals.values())
    return next(first[k] for k in first if totals[k] == best)
```

</details>

<details><summary>Solución: adaptive_vote</summary>

```python
from collections import Counter
from m09_reasoning.scaling import answer_key

def adaptive_vote(draw, max_n, lead):
    counts: Counter[str] = Counter()
    first: dict[str, str] = {}
    used = 0
    while used < max_n:
        answer = draw()
        used += 1
        key = answer_key(answer)
        if key is None or answer is None:
            continue
        counts[key] += 1
        first.setdefault(key, answer)
        top = counts.most_common(2)
        runner_up = top[1][1] if len(top) > 1 else 0
        if top[0][1] - runner_up >= lead:
            break
    if not counts:
        return None, used
    best = max(counts.values())
    return next(first[k] for k in first if counts[k] == best), used
```

</details>

## 💥 Rómpelo

Cada experimento cambia una línea. Ejecuta el comando, compara con la tabla de arriba y después
deshaz el cambio (`git checkout -- src` lo restaura todo). Los tests de `test_documented_numbers.py`
fallarán mientras el cambio siga aplicado: es lo esperado.

1. **Quita las trampas.** En `simulator.py`, pon a `0.0` los valores `order_total`, `energy` y
   `hub_capacity` de `BASE_TRAP` y ejecuta `uv run lab scale`. ¿Qué le pasa al estancamiento?

   <details><summary>Qué observar</summary>

   El estancamiento desaparece: con solo deslices independientes, la votación por mayoría alcanza a
   pass@N. Los errores correlacionados eran la única razón por la que la votación se atascaba.

   </details>

2. **Vuelve paranoica la autocrítica.** En `SimConfig`, intercambia los valores de `self_detect_slip`
   y `self_false_alarm`, y ejecuta `uv run lab revise`.

   <details><summary>Qué observar</summary>

   La fila de autocrítica apenas se mueve de la ronda 0 y el recuento "right->wrong" crece: un revisor
   que señala los pasos correctos más a menudo que los erróneos devuelve el buen trabajo para que lo
   estropeen. Las otras dos filas no cambian, porque no usan la crítica del modelo.

   </details>

3. **Sube la explotación de la recompensa.** En `experiments.py`, cambia `HACK_BIAS` de `5.0` a
   `10.0`, después a `0.0`, y ejecuta `uv run lab search` cada vez.

   <details><summary>Qué observar</summary>

   Con `10.0`, la columna sesgada se queda cerca de la exactitud de una sola muestra sea cual sea N:
   ahora la puntuación mide sobre todo la longitud. Con `0.0`, las dos columnas son idénticas. La
   presión de selección va hacia donde apunte la preferencia del verificador.

   </details>

4. **Deja que las pistas filtren las respuestas.** En `SimConfig`, pon `hint_lucky` a `0.9` y ejecuta
   `uv run lab star`, y después `uv run lab star --strict`.

   <details><summary>Qué observar</summary>

   La ejecución por defecto conserva más justificaciones infieles (respuesta correcta copiada de la
   pista, pasos erróneos), y la columna "unfaithful kept" crece. La ejecución estricta sigue sin
   conservar ninguna: comprobar los pasos, no solo las respuestas, es lo que protege los datos de
   entrenamiento.

   </details>

5. **Grupos de uno.** Ejecuta `uv run lab rl --group 1`.

   <details><summary>Qué observar</summary>

   Cada grupo tiene una sola muestra, así que todas las ventajas son cero: los 80 grupos aparecen "sin
   señal" y la política nunca se mueve. Los métodos relativos al grupo necesitan varias muestras por
   pregunta que discrepen entre sí.

   </details>

## 🚀 Amplíalo

1. **Combina verificadores.** Añade un método a `run_search` en `experiments.py`: conserva solo las
   muestras que cumplen `first_bad_step(...) is None` y elige después la mejor de ellas con el PRM.

   <details><summary>Pista</summary>

   Copia el bucle `filtered` de `run_search`, pero en lugar de `majority_vote` llama a `best_of_n(kept,
   [prm.score(p, a.steps) for a in kept])`. Añade una `row(...)` para él y compáralo con "best-of-9,
   PRM". Pregúntate si la recomprobación elimina algo que el PRM habría conservado.

   </details>

2. **Votación ponderada en la tabla de escalado.** Añade una columna a `run_scale` y `cmd_scale` con tu
   `weighted_vote` usando las puntuaciones del ORM.

   <details><summary>Pista</summary>

   En el bucle sobre `problems`, calcula `weighted_vote([a.answer for a in subset], [orm.score(p, a) for a
   in subset])` y añádelo a un campo nuevo de `ScaleRow`. Espera un resultado entre la votación simple y
   *best-of-N*: usa el verificador, pero sigue premiando las respuestas frecuentes.

   </details>

3. **Guarda en caché el evaluador del ToT.** La BFS a menudo valora dos veces el mismo conjunto de
   números (al que se llega con movimientos distintos). Añade una caché en `tot_bfs` indexada por los
   valores ordenados de un estado y cuenta las llamadas que ahorra.

   <details><summary>Pista</summary>

   Mantén `seen: dict[tuple[Fraction, ...], str] = {}` dentro de `tot_bfs`; antes de llamar a
   `reasoner.value(...)`, busca `tuple(sorted(v for v, _ in child))`. Compara la columna "calls" de
   `uv run lab tot` antes y después. Con un modelo real, cada llamada ahorrada es latencia y dinero.

   </details>

4. **Deja que el RL combata las trampas directamente.** Dale a `grpo_toy` un tercer parámetro por
   familia de problemas que escale la probabilidad de trampa, y apréndelo a partir de las decisiones de
   trampa de `sim.last_trace.trap_decisions`.

   <details><summary>Pista</summary>

   Añade un campo `trap_factor` a `SimConfig` y multiplícalo dentro de `trap_probability`. Para una
   probabilidad de trampa `q = q0 * 2 * sigmoid(theta)`, el gradiente de la log-probabilidad es
   `1 - sigmoid(theta)` cuando se cayó en la trampa y `-q * (1 - sigmoid(theta)) / (1 - q)` cuando se
   evitó. Compara la columna de *tokens*: ¿sigue aprendiendo la política a comprobar cuando puede
   simplemente dejar de caer en las trampas?

   </details>

## ✅ Definición de terminado (*Definition of Done*)

- [ ] `uv run pytest` pasa (54 tests, más los 3 tests de los ejercicios cuando los implementes).
- [ ] Sabes explicar qué es el razonador simulado, qué no es y por qué lo usa el lab.
- [ ] Sabes explicar a partir de `lab solve --id r076` qué error detectó o se le escapó a cada modo.
- [ ] Sabes explicar por qué se estanca la votación por mayoría y qué necesita *best-of-N* que la votación no necesita.
- [ ] Sabes comparar los tres bucles de revisión, incluidos sus cambios de correcta a errónea.
- [ ] Has seguido paso a paso una búsqueda ToT y una búsqueda en haz con PRM, y sabes explicar cada una de sus decisiones.
- [ ] Has ejecutado STaR en sus tres variantes y el juguete de RL, y sabes explicar la columna de grupos sin señal.
- [ ] Has implementado al menos dos ejercicios y probado al menos dos experimentos de Rómpelo.

## 🧯 Solución de problemas

| Síntoma | Causa y solución |
|---|---|
| `uv: command not found` | Instala uv desde https://docs.astral.sh/uv/ y abre un terminal nuevo. |
| `Could not find the repo's datasets/ folder` | Ejecuta el lab dentro del repositorio del curso, o define `COURSE_DATASETS_DIR` con la ruta de la carpeta `datasets`. |
| `The simulator only knows the problems in datasets/reasoning/problems.jsonl` | El simulador solo responde a las preguntas del dataset. Usa `make_reasoner()` con problemas de `load_problems()`. |
| `The simulator can only continue solutions that it wrote itself` | `next_step`, `revise` y `critique` necesitan pasos que haya producido el simulador. No edites los pasos antes de devolverlos. |
| Los números no coinciden con las tablas de las lecciones | Cambiaste un mando de `SimConfig`, de `experiments.py` o de un algoritmo. `git diff src` muestra qué; `git checkout -- src` lo restaura. |
| `test_documented_numbers.py` falla tras tu cambio | Es lo esperado mientras siga aplicado un cambio de Rómpelo o de Amplíalo. |

## 🔌 Usar un modelo real

La capa de proveedores puede enviar los mismos *prompts* a un modelo real. **Esto no se ejecutó al
escribir el lab:** Ollama estaba instalado pero no en marcha, y no se usó ninguna clave de API, así que
los comandos de abajo no están probados. Tómalos como punto de partida.

```bash
# Local model with Ollama (https://ollama.com): start the server and pull a model first.
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:1b uv run lab solve --provider ollama --mode cot --limit 20
uv run lab solve --provider ollama --id r076 --mode cot

# Any OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp server, a hosted API).
LLM_PROVIDER=openai OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_MODEL=your-model \
  uv run lab solve --provider openai --mode cot --limit 20
```

Qué esperar y qué no se traslada:

- Solo `solve` acepta `--provider`. Envía los *prompts* de `prompts.py` y califica la línea `Answer:`
  con las mismas comprobaciones exactas; los modelos pequeños a menudo ignoran el formato, lo que
  aparece como respuestas erróneas.
- Si un modelo escribe su razonamiento entre etiquetas `<think>`, `parse_solution` lo cuenta como
  *tokens* de pensamiento. Los modelos de razonamiento alojados suelen ocultar su razonamiento; en ese
  caso usa los recuentos de *tokens* de los datos de uso del proveedor (el objeto `Completion` guarda
  la respuesta en bruto en `raw`).
- El ORM y el PRM simulados, la autocrítica, la búsqueda en haz por pasos, STaR y el juguete de RL
  dependen de la contabilidad interna del simulador o de sus mandos, así que solo funcionan con el
  proveedor falso por defecto. Para usar `scaling.py` o `search.py` con un modelo real, construye un
  `Reasoner(get_provider("ollama"))` y llama directamente a las funciones; la votación, *best-of-N* con
  tu propio puntuador, el muestreo hasta verificar y el árbol de pensamientos solo necesitan la
  interfaz de chat.

---

[← Anterior: 9.4 Entrenar modelos de razonamiento](../04-training-reasoning-models.es.md) · [🏠 Módulo 09](../README.es.md) · [Siguiente: Quiz →](../quiz.es.md)
