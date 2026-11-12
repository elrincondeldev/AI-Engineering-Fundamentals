🌐 [English](README.md) · Español

# Lab 04 — Un enrutador de intenciones, adaptado de tres formas

> Enruta los tickets de soporte de Hearthwise por intención con *prompts* sin ejemplos y con pocos
> ejemplos, un ensamblador de contexto con presupuesto y *fine-tuning* con LoRA, todo puntuado con una
> partición temporal honesta. ⏱️ unas 2,5 horas.

## 🎯 Qué vas a construir

Alex quiere que cada ticket entrante vaya a la cola correcta entre ocho intenciones. Dani quiere que
cada afirmación esté respaldada por una evaluación que no se pueda engañar a sí misma. Vas a
construir:

- una **auditoría de datos** que encuentra duplicados, datos personales, una columna con fuga y deriva
  de concepto;
- un **arnés de evaluación** con una partición temporal, exactitud por segmento (español, cámara) y
  coste en *tokens*;
- un **constructor de *prompts*** con definiciones de etiquetas, selección de ejemplos *few-shot*,
  contexto de la persona usuaria, salida JSON, validación y reintentos;
- un **ensamblador de contexto** que encaja en un presupuesto de *tokens*, compacta el texto e informa
  de lo que descartó, más una simulación de caché de prefijo;
- **LoRA desde cero en numpy**, comparado con el *fine-tuning* completo, con una prueba de olvido y
  una curva según el tamaño de los datos;
- cuatro pequeñas funciones tuyas en `src/m04_adaptation/exercises.py`.

El "LLM" por defecto es un **aprendiz en contexto de juguete** (*toy in-context learner*, `icl.py`)
conectado al `FakeProvider` del curso. Lee las definiciones de las etiquetas y los ejemplos del
*prompt* y vota según las palabras que coinciden, así que cada respuesta se puede explicar y cada
ejecución es idéntica. No es un LLM real: úsalo para construir y depurar el arnés, y después apunta el
mismo arnés a un modelo real (consulta la última sección).

| Archivo | Qué hace |
|---|---|
| `tickets.py` | Carga, limpia, anonimiza, deduplica y parte los tickets |
| `textfeat.py` | Raíces de palabras, similitud de Jaccard, un *hash* estable |
| `icl.py` | El aprendiz en contexto de juguete (un `responder` de `FakeProvider`) |
| `prompts.py` | Definiciones de etiquetas, `PromptConfig`, selección de ejemplos, construcción de mensajes |
| `structured.py` | Analiza y valida las respuestas JSON, reintenta una vez con el error |
| `evaluation.py` | Exactitud por segmento, línea base de la clase mayoritaria, auditoría de fugas columna a columna |
| `context.py` | Piezas de contexto, ensamblado con presupuesto, orden, deduplicación, caché de prefijo |
| `lora.py` | Variables con *hashing*, clasificador *softmax*, *fine-tuning* completo, LoRA, recuentos de parámetros |
| `reports.py` | Todos los experimentos citados en las lecciones (comprobados en `tests/test_documented_numbers.py`) |
| `exercises.py` | Tus cuatro funciones (lanzan `NotImplementedError` hasta que las escribas) |
| `providers.py`, `data.py` | Copiados sin cambios de `tools/lab_skeleton` |

## ⚙️ Preparación

Desde la raíz del repositorio:

```bash
cd modules/04-prompting-context-and-fine-tuning/lab
uv sync
uv run pytest
```

Resumen esperado (el tiempo es un ejemplo y varía según la máquina):

```text
81 passed, 5 skipped in 16.18s
```

Cuatro de los tests omitidos corresponden a los ejercicios que todavía no has escrito; el quinto es un
test con un modelo real que solo se ejecuta con `RUN_LIVE_TESTS=1`. Lista los comandos con
`uv run lab --help`.

## 👣 Pasos

### Paso 1 — Audita los tickets

Lección: [4.1](../01-adaptation-techniques.es.md). Cuenta los duplicados y los datos personales, busca
una columna que conozca la respuesta y mide la deriva tras el lanzamiento de la Lookout C2.

```bash
uv run lab audit
```

```text
Ticket audit
------------
rows: 1224 · resubmitted copies: 24 · repeated bodies (any customer): 51
after removing resubmissions: 1200 · temporal split: train 800 (before 2025-09-01) / test 400
personal data and noise in bodies: phone_number 82, email_address 0, signature 205, quoted_reply 142, html_entity 23

Can one column predict the intent? (rule learned on train, scored on test)
--------------------------------------------------------------------------
column           values  accuracy  known when the ticket arrives?
channel               3     0.120  yes
language              2     0.125  yes
product               5     0.125  yes
customer_email      763     0.113  yes
resolution_code       8     1.000  NO  <- leak: filled in after the ticket is solved
majority-class baseline on test: 0.120

Drift after the Lookout C2 launch
---------------------------------
intent                before   after
order_status           0.181   0.120
return_request         0.155   0.095
warranty_claim         0.081   0.087
connectivity_issue     0.139   0.278
billing_question       0.133   0.120
account_access         0.111   0.110
product_question       0.138   0.125
cancel_subscription    0.062   0.065
Lookout C2 share of test tickets: 0.395
unseen word stems vs Jan-Jul vocabulary: August 0.000 · after launch 0.186 · camera tickets 0.465
```

Lee `find_resubmissions` y `clean_text` en `tickets.py`, y `single_column_accuracy` en
`evaluation.py`, antes de seguir.

### Paso 2 — Compara particiones de evaluación

Lección: [4.1](../01-adaptation-techniques.es.md). El mismo *prompt*, puntuado con cuatro diseños de
evaluación y una vez más con el campo con fuga pegado en el *prompt*.

```bash
uv run lab splits
```

```text
Same prompt (bilingual descriptions + 8 similar examples), different evaluation designs
---------------------------------------------------------------------------------------
split                          train  test overlap accuracy
random 80/20, raw rows           979   245      16    0.959
random 80/20, deduplicated       960   240       6    0.921
temporal, raw rows               814   410      13    0.820
temporal, deduplicated           800   400      11    0.828
temporal + resolution_code       800   400      11    0.920
overlap = test tickets whose exact text also appears in train
```

**Te toca (ejercicio 1).** Implementa `flag_leaky_columns(audits, threshold)` en `exercises.py`:
devuelve las columnas que no se conocen en el momento de predecir o que predicen la etiqueta por sí
solas con una exactitud igual o superior al umbral. Hasta que la escribas, el test se omite; cuando
sea correcta verás (los tiempos varían):

```bash
uv run pytest tests/test_exercises.py -k leaky
```

```text
.                                                                        [100%]
1 passed, 3 deselected in 0.14s
```

<details><summary>Solución de referencia</summary>

```python
def flag_leaky_columns(audits: list[ColumnAudit], threshold: float = 0.95) -> list[str]:
    return [a.column for a in audits if not a.known_at_prediction_time or a.accuracy >= threshold]
```

</details>

### Paso 3 — Lee un prompt de enrutamiento

Lección: [4.2](../02-prompt-engineering.es.md). Imprime todos los mensajes que envía el enrutador para
el ticket de la cámara mal enrutado de Alex (T00843) con dos ejemplos parecidos, más la respuesta en
bruto del modelo.

```bash
uv run lab show-prompt
```

La salida empieza con el mensaje de sistema completo y termina con:

```text
[user]
Ticket: Hey, ¿El Glow A19 es compatible con el wifi de 5 GHz? Thank you!

[assistant]
{"intent": "product_question"}

[user]
Ticket: Buenas, ¿El Glow A19 es compatible con el wifi de 5 GHz? Urgent!

[assistant]
{"intent": "product_question"}

[user]
Ticket: Buenas, La cámara se queda en 'conectando' desde que la pasé a 5 GHz. Thanks.

[model answer]
{"intent": "product_question"}

-> predicted: product_question · true: connectivity_issue · attempts: 1 · prompt tokens: 415
```

Prueba otras opciones: `--k 0`, `--selection random`, `--user-context`, `--reasoning`,
`--descriptions names` y `--descriptions bilingual_v2`, y `--ticket` con el ID de cualquier ticket de
prueba.

### Paso 4 — Compara estrategias de prompting

Lección: [4.2](../02-prompt-engineering.es.md). La tabla de Dani: una fila por variante de *prompt*.

```bash
uv run lab prompts
```

```text
Prompting strategies on the temporal test set (provider: fake)
--------------------------------------------------------------
prompt                                acc  camera spanish   tokens
label names only, 0-shot            0.270   0.228   0.125    130.5
English descriptions, 0-shot        0.618   0.443   0.333    276.1
bilingual descriptions, 0-shot      0.723   0.519   0.683    350.9
bilingual + 1 random example        0.728   0.519   0.692    389.0
bilingual + 4 random examples       0.735   0.525   0.708    504.3
bilingual + 8 random examples       0.728   0.532   0.692    656.8
bilingual + 8 balanced examples     0.740   0.557   0.708    654.3
bilingual + 1 similar example       0.782   0.595   0.750    388.8
bilingual + 4 similar examples      0.828   0.633   0.775    502.6
bilingual + 8 similar examples      0.828   0.633   0.775    654.0
8 similar + user context            0.845   0.677   0.833    790.5
n = 400 tickets · tokens = mean prompt tokens per ticket
```

**Te toca (ejercicio 2).** La selección por similitud a menudo devuelve casi copias del mismo ticket.
Implementa `select_diverse` con relevancia marginal máxima (MMR, *maximal marginal relevance*):
prefiere los candidatos que se parecen a la consulta pero no a los ejemplos ya elegidos.

```bash
uv run pytest tests/test_exercises.py -k diverse
```

```text
.                                                                        [100%]
1 passed, 3 deselected in 0.16s
```

<details><summary>Solución de referencia</summary>

```python
def select_diverse(selector: ExampleSelector, query: Ticket, k: int, diversity: float = 0.7) -> list[Ticket]:
    from .prompts import user_message
    from .textfeat import jaccard, stems

    candidates = selector.select(query, 4 * k, "similar")[::-1]  # most similar first
    feats = {t.ticket_id: stems(user_message(t, selector.cfg)) for t in candidates}
    q = stems(user_message(query, selector.cfg))
    chosen: list[Ticket] = []
    while candidates and len(chosen) < k:
        def mmr(c: Ticket) -> float:
            redundancy = max((jaccard(feats[c.ticket_id], feats[s.ticket_id]) for s in chosen), default=0.0)
            return (1 - diversity) * jaccard(q, feats[c.ticket_id]) - diversity * redundancy
        best = max(candidates, key=mmr)
        chosen.append(best)
        candidates.remove(best)
    return chosen
```

</details>

### Paso 5 — Valida la salida estructurada

Lección: [4.2](../02-prompt-engineering.es.md). ¿Con qué frecuencia rompe la primera respuesta el
formato JSON, qué recuperan un analizador tolerante y un reintento, y cuánto cuesta el *prompting*
paso a paso?

```bash
uv run lab format
```

```text
Structured output on 400 test tickets (bilingual + 8 similar, JSON)
-------------------------------------------------------------------
first answer is strict, valid JSON:        379
first answer readable by a lenient parser: 390
needed a retry:                            10
still invalid after one retry:             0
accuracy without retry: 0.810 · with one retry: 0.828
step-by-step prompt: accuracy 0.828 · completion tokens 9.4 -> 35.7
```

Lee `parse_answer` y `classify` en `structured.py`: son el código que mantendrías sin cambios al pasar
a un modelo real.

### Paso 6 — Ensambla el contexto con un presupuesto

Lección: [4.3](../03-context-engineering.es.md). Encaja el T00822 en el presupuesto de 500 *tokens* de
Sam y lee el informe de descartes; después barre presupuestos y políticas de compactación.

```bash
uv run lab context
uv run lab context --policy raw
uv run lab budget
```

El primer comando imprime las piezas conservadas y descartadas (la salida completa está en la lección
4.3); sus primeras líneas son:

```text
Context for T00822 · budget 500 tokens · policy clean+dedupe
------------------------------------------------------------
offered 884 tokens · sent 487 tokens · examples kept 4
```

Con `--policy raw`, en el mismo presupuesto cabe un solo ejemplo:

```text
Context for T00822 · budget 500 tokens · policy raw
---------------------------------------------------
offered 1351 tokens · sent 447 tokens · examples kept 1
```

El barrido de presupuestos:

```text
Token budget sweep (16 candidate examples, profile, query; 400 test tickets)
----------------------------------------------------------------------------
budget policy        accuracy examples  tokens
   400 raw              0.790     0.71   388.4
   400 clean            0.797     0.89   390.5
   400 clean+dedupe     0.797     0.86   389.6
   500 raw              0.820     3.21   485.9
   500 clean            0.828     3.50   486.0
   500 clean+dedupe     0.830     3.26   478.3
   700 raw              0.815     8.17   684.4
   700 clean            0.843     8.96   685.1
   700 clean+dedupe     0.830     6.07   583.0
  1000 raw              0.807    14.64   957.4
  1000 clean            0.843    15.67   933.2
  1000 clean+dedupe     0.830     6.28   592.0

More examples are not always better
-----------------------------------
prompt                                acc  camera spanish   tokens
4 similar examples                  0.828   0.633   0.775    502.6
8 similar examples                  0.828   0.633   0.775    654.0
16 similar examples                 0.840   0.665   0.792    955.0
32 similar examples                 0.825   0.639   0.783   1555.6

Tickets with a quoted earlier e-mail (n = 47)
---------------------------------------------
0-shot: raw 0.681 -> cleaned 0.809
8-shot: raw 0.596 -> cleaned 0.830
```

**Te toca (ejercicio 3).** Implementa `truncate_middle(text, max_tokens)`: conserva el principio y el
final de un texto demasiado largo y corta el medio, marcando el corte con `[...]`.

```bash
uv run pytest tests/test_exercises.py -k truncate
```

```text
.                                                                        [100%]
1 passed, 3 deselected in 0.14s
```

<details><summary>Solución de referencia</summary>

```python
def truncate_middle(text: str, max_tokens: int, marker: str = " [...] ") -> str:
    from .providers import count_tokens

    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    for keep in range(len(words) - 1, 1, -1):
        head, tail = words[: (keep + 1) // 2], words[len(words) - keep // 2 :]
        candidate = " ".join(head) + marker + " ".join(tail)
        if count_tokens(candidate) <= max_tokens:
            return candidate
    return marker.strip()
```

</details>

### Paso 7 — Guarda en caché el prefijo estable

Lección: [4.3](../03-context-engineering.es.md). Vuelve a reproducir las 400 peticiones de prueba a
través de una caché de prefijo simulada, una vez con un *prompt* de sistema estable y otra con un ID
de petición y una fecha en su primera línea.

```bash
uv run lab cache
```

```text
Prefix cache simulation over the 400 test requests
--------------------------------------------------
stable system prompt     hits 399 · misses   1 · hit rate 0.998 · cached share of prompt tokens 0.496
request id + date first  hits   0 · misses 400 · hit rate 0.000 · cached share of prompt tokens 0.000
```

### Paso 8 — Adapta el enrutador a la deriva sin entrenar

Lección: [4.4](../04-fine-tuning-peft-and-lora.es.md). A partir de aquí los meses se reparten en
cuatro partes: enero–julio (704 tickets) para el banco de ejemplos o el modelo base, agosto (96) como
prueba "antigua", septiembre (97) para la adaptación y octubre–diciembre (303) como prueba "nueva".

```bash
uv run lab drift
```

```text
Prompt-side adaptation (old = August, new = October-December)
-------------------------------------------------------------
approach                         labelled    old    new  camera  tokens
0-shot, original descriptions           0  0.760  0.710   0.496   352.7
0-shot, descriptions updated            0  0.760  0.851   0.851   430.9
8 similar, pool Jan-Jul               704  0.917  0.825   0.620   656.0
8 similar, pool Jan-Jul + Sep         801  0.917  0.901   0.810   658.7
```

Las definiciones actualizadas son `CAMERA_UPDATE` en `prompts.py`. Léelas y pregúntate qué tickets de
octubre se les escaparían.

### Paso 9 — Haz fine-tuning con LoRA

Lección: [4.4](../04-fine-tuning-peft-and-lora.es.md). Entrena un clasificador lineal de intenciones
con enero–julio, congélalo y adáptalo a septiembre con *fine-tuning* completo y con LoRA de rangos 1,
2, 4 y 8. Los resultados de numpy pueden diferir en aproximadamente un ticket en otra máquina; los
tests lo tienen en cuenta.

```bash
uv run lab lora
uv run lab lora-params
```

```text
Adapting the frozen classifier with 97 September tickets (LoRA: mean of 5 seeds)
--------------------------------------------------------------------------------
model                      trainable  share    old    new  camera  spread
base (frozen, no update)           0  0.000  0.917  0.828   0.628   0.000
full fine-tuning                8200  1.000  0.906  0.927   0.893   0.000
LoRA r=1                        1032  0.126  0.906  0.870   0.770   0.042
LoRA r=2                        2064  0.252  0.904  0.893   0.825   0.046
LoRA r=4                        4128  0.503  0.904  0.905   0.868   0.020
LoRA r=8                        8256  1.007  0.904  0.906   0.874   0.013

Forgetting: train longer on September only (old = August accuracy)
------------------------------------------------------------------
steps  full old  full new  r=4 old  r=4 new replay old replay new
  100     0.906     0.927    0.906    0.901      0.906      0.927
 1000     0.875     0.904    0.885    0.881      0.906      0.914
 4000     0.792     0.855    0.812    0.848      0.865      0.888

How many labelled September tickets? (accuracy on October-December)
-------------------------------------------------------------------
labelled   full  LoRA r=4
      12  0.865     0.867
      25  0.871     0.881
      50  0.911     0.908
      97  0.927     0.905
```

La salida de `lora-params` aparece en la lección 4.4. Lee `train_lora` en `lora.py` junto a las
matemáticas de la lección: cada línea corresponde a una ecuación.

**Te toca (ejercicio 4).** Implementa `lora_logits`: la pasada hacia delante de LoRA que nunca
construye la matriz de actualización `(C, D)`. Con matrices grandes, así es como funcionan los
adaptadores antes de fusionarlos.

```bash
uv run pytest tests/test_exercises.py -k lora
```

```text
.                                                                        [100%]
1 passed, 3 deselected in 0.13s
```

<details><summary>Solución de referencia</summary>

```python
def lora_logits(X, W, b, A, B, alpha):
    r = A.shape[0]
    return X @ W.T + b + (alpha / r) * (X @ A.T) @ B.T
```

`X @ A.T` tiene forma `(n, r)`, así que el camino de bajo rango cuesta `n·r·(D + C)`
multiplicaciones en lugar de construir antes una matriz de `C × D`.

</details>

### Paso 10 — Escribe la recomendación

Ejecuta toda la batería de tests una vez más. Con tus cuatro ejercicios hechos, no se omite nada
salvo el test con un modelo real:

```bash
uv run pytest
```

```text
85 passed, 1 skipped in 16.40s
```

Después escríbeles a Alex y Dani una recomendación breve (media página) en tu diario de progreso. Usa
este esquema y solo números que hayas obtenido tú:

```text
Decision: <prompt only | prompt now + LoRA adapter later | fine-tune now>
Evidence (new = Oct-Dec, old = August, camera = Lookout C2 tickets):
  - best prompt without training: ...
  - LoRA adapter (rank, trainable numbers): ...
  - forgetting check on August: ...
Cost per ticket: prompt tokens ... versus ...
Risks: label noise, drift after the next launch, personal data in tickets
What we monitor and what triggers retraining: ...
```

## 💥 Rómpelo

**1. Saca los ejemplos *few-shot* del conjunto de prueba.** Ejecuta esto en `uv run python`. La
exactitud salta de 0.828 a 0.943. ¿Por qué ese número no vale nada?

```python
from m04_adaptation.evaluation import evaluate_prompt
from m04_adaptation.reports import BEST_PROMPT, standard_split

train, test = standard_split()
print(evaluate_prompt("leaky pool", test, test, BEST_PROMPT).accuracy)  # pool = test set
```

<details><summary>Qué está pasando</summary>

Ahora el ejemplo más parecido a cada ticket es el propio ticket, etiquetado con su propia respuesta.
El *prompt* contiene la hoja de respuestas, exactamente igual que la fuga de `resolution_code` de la
lección 4.1. En producción, el banco solo puede contener tickets resueltos antes de que llegue el
nuevo. No llega a 1.000 porque las definiciones de las etiquetas siguen arrastrando algunos tickets
hacia otras intenciones.

</details>

**2. Cambia el orden de las intenciones en el *prompt* de sistema.** En `system_prompt`
(`prompts.py`), recorre `reversed(descriptions(cfg.descriptions).items())` y vuelve a ejecutar
`uv run lab prompts`. La fila bilingüe sin ejemplos pasa de 0.723 a 0.755 aunque no ha cambiado ni una
palabra del *prompt*.

<details><summary>Qué está pasando</summary>

Cuando dos intenciones empatan, el juguete elige la que aparece primero en la lista, y cuando no
coincide nada recurre a la primera intención. El orden de las etiquetas es un valor por defecto
oculto. Los modelos reales tienen sus propias preferencias de orden y de posición, así que trata el
orden de las etiquetas y de los ejemplos como una variable que hay que probar, no como decoración.

</details>

**3. Sube la escala de LoRA.** Ejecuta esto en `uv run python`. La exactitud en octubre–diciembre se
desploma hasta 0.122. Imprime `abs(model.B).max()` para ver por qué.

```python
from m04_adaptation.lora import train_lora
from m04_adaptation.reports import base_model, lora_data

d = lora_data()
model = train_lora(base_model(), d.X_adapt, d.y_adapt, rank=1, alpha=8)  # scale alpha/r = 8
print(model.accuracy(d.X_new, d.y_new))
```

<details><summary>Qué está pasando</summary>

La actualización es `(alpha/r)·B·A`, y se entrenan tanto `A` como `B`, así que una escala de 8
multiplica muchas veces el tamaño efectivo del paso. El descenso de gradiente simple se pasa de largo,
las entradas de `A` y `B` crecen sin límite y casi todos los tickets acaban con la misma etiqueta. Por
eso el lab mantiene `alpha = r` (escala 1), y por eso cambiar el rango sin pensar en alpha y en la
tasa de aprendizaje puede romper el entrenamiento.

</details>

**4. Deja de anonimizar los datos personales.** En `clean_text` (`tickets.py`), borra la línea
`text = redact_pii(text)` y ejecuta `uv run pytest`. ¿Qué test falla? ¿Cambia alguno de los números
del lab?

<details><summary>Qué está pasando</summary>

Falla `test_clean_text_removes_quotes_signatures_entities_and_pii`, porque un número de teléfono
escrito en el cuerpo de un ticket ahora sobrevive a la limpieza. Ninguno de los números documentados
cambia: en este dataset todos los teléfonos están en firmas, y las firmas ya se eliminan. Esa es la
cuestión. Un control de privacidad puede romperse sin mover ninguna métrica, así que solo lo detecta
un test unitario dedicado. Sin él, el siguiente ticket que diga "llámame al ..." llevaría el número a
los *prompts*, a los bancos de ejemplos y a los datos de entrenamiento del adaptador, que es
exactamente lo que Noor te pidió evitar en el módulo 03.

</details>

## 🚀 Amplíalo

**1. Construye una sonda de posición.** Mide si importa la posición del ejemplo más útil: para cada
ticket, toma siete ejemplos aleatorios más el único más parecido (la "aguja"), inserta la aguja en las
posiciones de la 0 a la 7 y calcula la exactitud por posición. Ejecútalo con
`ToyInContextLearner(position_bias=0.0)` (la línea debe salir plana) y con `position_bias=0.6`.

<details><summary>Pista y solución</summary>

```python
from m04_adaptation.evaluation import make_provider
from m04_adaptation.icl import ToyInContextLearner
from m04_adaptation.prompts import ExampleSelector, answer_message, build_messages, user_message
from m04_adaptation.providers import Message
from m04_adaptation.reports import BEST_PROMPT, standard_split
from m04_adaptation.structured import classify
from m04_adaptation.tickets import INTENTS

train, test = standard_split()
cfg, selector = BEST_PROMPT, ExampleSelector(train, BEST_PROMPT)
provider = make_provider(ToyInContextLearner(position_bias=0.6))
for pos in range(8):
    hits = 0
    for t in test:
        examples = selector.select(t, 7, "random")
        examples.insert(pos, selector.select(t, 1, "similar")[0])
        base = build_messages(t, cfg)  # [system, query]
        msgs = [base[0]]
        for ex in examples:
            msgs += [Message("user", user_message(ex, cfg)), Message("assistant", answer_message(ex.intent, cfg))]
        hits += classify(provider, msgs + [base[-1]], INTENTS).label == t.intent
    print(pos, round(hits / len(test), 3))
```

Con `position_bias=0`, todas las posiciones dan la misma exactitud, lo que valida la sonda. Con el
sesgo activado, la exactitud cambia con la posición, y la dirección puede sorprenderte: con peso
completo, un único ejemplo parecido a veces se impone a unas definiciones de etiquetas correctas. El
trabajo de una sonda es detectar la sensibilidad; ejecútala con un modelo real para conocer la forma
de ese modelo.

</details>

**2. Deja que el enrutador se abstenga.** Un ticket mal enrutado cuesta más que un ticket enviado al
triaje humano. Después de cada llamada, lee `learner.last_scores` (crea
`learner = ToyInContextLearner()` y pásalo a `make_provider(learner)`), calcula el margen entre la
mejor y la segunda mejor puntuación y envía a una persona los tickets que queden por debajo de un
umbral. Informa de la cobertura y de la exactitud en los tickets que conservaste.

<details><summary>Pista</summary>

Ordena `learner.last_scores.values()` de mayor a menor y calcula `top[0] - top[1]`. Barre umbrales
como 0, 0,5, 1 y 2. La exactitud en los tickets conservados debería subir mientras baja la cobertura;
el umbral adecuado es una decisión de negocio para Alex (cuánto trabajo de triaje es aceptable), no
una decisión técnica.

</details>

**3. Barre el tamaño de la repetición.** En `forgetting_table`, la muestra de repetición tiene 100
tickets antiguos. Prueba 0, 25, 50, 100 y 200 con 1.000 pasos y dibuja la exactitud de agosto frente
a la de octubre–diciembre. ¿Hasta qué punto puede reducirse la muestra de repetición antes de que
vuelva el olvido?

<details><summary>Pista</summary>

Copia el cuerpo de `forgetting_table` en un script, recorre en un bucle el tamaño de la muestra y
mantén fija la semilla para que lo único que cambie sea el tamaño. Informa de los dos conjuntos de
prueba para cada tamaño: una repetición que protege agosto pero perjudica octubre–diciembre no sale
gratis.

</details>

**4. Dale a LoRA una matriz en la que compense.** Añade una capa oculta (1.024 entradas → 64
unidades ocultas → 8 intenciones) al clasificador, preentrénalo, congélalo y aplica LoRA a la matriz
de 64 × 1.024. Compara los parámetros entrenables y la exactitud con el *fine-tuning* completo de las
dos capas.

<details><summary>Pista</summary>

La pasada hacia delante es `h = relu(X @ W1.T + b1)`, `logits = h @ W2.T + b2`. Retropropaga la señal
de error a través de `W2` y de la máscara de la ReLU para obtener el gradiente de `W1`, y después
aplica las mismas dos líneas que `train_lora` para obtener los gradientes de `A` y `B`. Con
`d_in = 1,024` y `d_out = 64`, el rango 4 entrena `4 · (1,024 + 64)` números frente a 65.536 de la
matriz completa, así que esta vez el ahorro es real. Añade una comprobación del gradiente por
diferencias finitas (consulta `tests/test_lora.py`) antes de fiarte de ningún resultado.

</details>

## ✅ Definición de terminado (*Definition of Done*)

- [ ] `uv run pytest` pasa y, después de tus ejercicios, informa de 85 tests superados y 1 omitido.
- [ ] Implementaste las cuatro funciones de `exercises.py` sin copiar las soluciones de referencia.
- [ ] Sabes explicar por qué el mismo *prompt* obtiene 0.959 con una partición aleatoria y 0.828 con
      la temporal.
- [ ] Sabes nombrar la columna con fuga, mostrar la evidencia y enumerar todos los caminos por los que
      podría entrar en un *prompt*.
- [ ] Reprodujiste la tabla de *prompts* de Dani y sabes decir qué variante lanzarías y cuánto cuesta
      en *tokens* de *prompt*.
- [ ] Sabes leer el informe de descartes del T00822 y explicar cada línea "near-duplicate" y "over
      budget".
- [ ] Comparaste la adaptación por el lado del *prompt* con LoRA en tickets nuevos *y* antiguos, y
      escribiste la recomendación del paso 10 con tus propios números.
- [ ] Completaste al menos un experimento de "Rómpelo" y una ampliación de "Amplíalo".

## 🧯 Solución de problemas

| Síntoma | Solución |
|---|---|
| `uv: command not found` | Instala uv desde [docs.astral.sh/uv](https://docs.astral.sh/uv/) y abre un terminal nuevo. |
| `FileNotFoundError: Could not find the repo's datasets/ folder` | Ejecuta el lab desde dentro del repositorio, o define `COURSE_DATASETS_DIR` con la ruta de la carpeta `datasets`. |
| `lab: command not found` | Usa `uv run lab ...` desde la carpeta `lab`, para que uv active el entorno. |
| Los números de LoRA difieren de la lección en unos 0.01 | Es esperable en algunas máquinas: otra compilación de numpy/BLAS puede cambiar una predicción en el límite. Los tests admiten 0.011. |
| La exactitud se desploma o aparece `RuntimeWarning: overflow encountered` | El entrenamiento divergió: baja la tasa de aprendizaje o mantén `alpha = rank` (consulta Rómpelo 3). |
| Un número de *prompting* difiere de la lección | Los resultados de *prompting* son Python puro y exactos. Comprueba que no editaste `prompts.py`, `icl.py` ni los datasets, y que `LLM_PROVIDER` no está definida. |
| Un test de `test_exercises.py` falla en lugar de omitirse | Tu función ya no lanza `NotImplementedError`, pero devuelve un resultado incorrecto. Lee el mensaje de la aserción. |

## 🔌 Usar un modelo real

Todo lo anterior funciona sin conexión con el aprendiz de juguete. La capa de proveedores
(`providers.py`, copiada del esqueleto del curso) también puede hablar con un modelo local a través de
Ollama o con cualquier *endpoint* compatible con OpenAI, y hay dos comandos pensados para ello:

```bash
# Local model with Ollama (https://ollama.com), after `ollama pull llama3.2:1b`
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:1b uv run lab show-prompt --k 4
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:1b uv run lab prompts --limit 40

# Any OpenAI-compatible endpoint (a hosted API, vLLM, LM Studio, llama.cpp server)
LLM_PROVIDER=openai OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_MODEL=<model> uv run lab prompts --limit 40

# The optional live test (3 tickets through Ollama)
RUN_LIVE_TESTS=1 uv run pytest -m ollama
```

Con un proveedor real, `prompts` puntúa solo los N primeros tickets de prueba (40 por defecto), porque
11 variantes de *prompt* por 400 tickets son muchas llamadas. Los comandos `splits`, `format`, `budget`
y `drift` reproducen cientos de llamadas por configuración, así que siempre usan el aprendiz de
juguete y lo indican.

Ten claro qué se ha comprobado y qué no:

- **No probado aquí.** Estos caminos con modelos reales no se ejecutaron cuando se escribió este lab.
  Ollama estaba instalado pero no en marcha, y no se usó ninguna clave de API. Los comandos de arriba
  pasan por el mismo `providers.py` que usan los demás labs, y los tests sin conexión comprueban que
  las peticiones se construyen correctamente, pero no se ha verificado ninguna salida de un modelo
  real.
- **Espera números distintos.** Un LLM real entiende los nombres de las etiquetas, los sinónimos y el
  español mucho mejor que el juguete, así que el *zero-shot* será mucho más fuerte y las diferencias
  entre filas cambiarán. No fallará en el formato siguiendo un calendario fijo, puede ignorar u
  obedecer la instrucción de pensar paso a paso, y sus puntuaciones variarán con el modelo y su
  versión. De eso se trata: el arnés sigue siendo el mismo y te dice qué hace *tu* modelo.
- **Cuidado con los datos.** Los *prompts* contienen texto de tickets limpio y con los datos
  personales anonimizados. Antes de enviarlos a una API externa, comprueba con la persona responsable
  de privacidad (Noor, en nuestra historia) que está permitido.
- **LoRA sobre un LLM real** necesita una GPU y una biblioteca como Hugging Face PEFT; queda fuera de
  este lab. La aritmética de parámetros de `lora-params` se aplica sin cambios.

---

[← Anterior: 4.4 *Fine-tuning*, PEFT y LoRA](../04-fine-tuning-peft-and-lora.es.md) · [🏠 Módulo 04](../README.es.md) · [Siguiente: Quiz →](../quiz.es.md)
