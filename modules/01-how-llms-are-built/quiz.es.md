🌐 [English](quiz.md) · Español

# Quiz del módulo 01 — Cómo se construyen los LLM

Responde cada pregunta antes de abrir la solución. Apunta tus respuestas; explicar el *porqué* cuenta
tanto como la letra. Las preguntas marcadas con 🔁 repasan la primera lección de este módulo.

### P1

En cada paso de la generación, ¿qué produce un modelo de lenguaje grande?

- A. La siguiente frase completa
- B. Una probabilidad para cada *token* de su vocabulario
- C. El documento más relevante de sus datos de entrenamiento
- D. Una sola palabra elegida de un diccionario

<details><summary>Respuesta</summary>

**B.** El modelo produce una distribución sobre todo su vocabulario; se elige un *token* (de forma
voraz o muestreando), se añade y el bucle se repite. No recupera documentos (C), y los *tokens* suelen
ser trozos de palabras, no palabras de diccionario (D).

</details>

### P2

¿Qué formato de Common Crawl contiene texto plano ya extraído de las páginas?

- A. WARC
- B. WAT
- C. WET
- D. JSONL

<details><summary>Respuesta</summary>

**C. WET.** WARC contiene las respuestas HTTP en bruto y WAT contiene metadatos como los enlaces. Los
*pipelines* centrados en la calidad suelen partir de WARC y hacer su propia extracción, porque el texto
extraído de forma genérica conserva el texto repetitivo (*boilerplate*).

</details>

### P3

El *pipeline* del lab informa de `TP=82 FP=0 FN=1 TN=51`, con "conservar" como clase positiva.
Calcula la precisión y el *recall*.

<details><summary>Respuesta</summary>

Precisión = TP / (TP + FP) = 82 / 82 = **1.000**. *Recall* = TP / (TP + FN) = 82 / 83 = **0.988**.
Todas las páginas conservadas merecían conservarse; una página que merecía conservarse (la ficha de
producto del Corvid Thermo X) se descartó.

</details>

### P4

Tu filtro de repetición descarta fichas de producto cortas que repiten el nombre del producto. ¿Qué
harías?

<details><summary>Respuesta</summary>

Mira primero las páginas que fallan (el lab las lista). Después cambia la regla de forma acotada:
aplica la comprobación de repetición solo a partir de una longitud mínima, o ignora las repeticiones
del título. Vuelve a ejecutar y comprueba que el *recall* mejora sin dejar pasar *spam* (precisión).
Por último, confírmalo con una muestra etiquetada con la que no hayas ajustado nada, porque ajustar y
puntuar con las mismas páginas embellece el resultado.

</details>

### P5

Encuentra el fallo. Esta función auxiliar de MinHash da firmas distintas cada vez que arranca el
programa, así que el índice de ayer no se puede comparar con el de hoy.

```python
def shingle_ids(items: set[str]) -> list[int]:
    return [hash(s) % PRIME for s in items]
```

<details><summary>Respuesta</summary>

El `hash()` integrado de Python para cadenas usa una semilla aleatoria distinta en cada proceso, así
que la misma teja recibe un entero distinto en cada ejecución. Usa un *hash* estable como
`hashlib.blake2b` o SHA-256, como hace el `_shingle_ids` del lab. Ordenar el conjunto antes de
calcular los *hashes* también mantiene determinista el orden de la salida.

</details>

### P6

Un tokenizador BPE a nivel de byte se entrena con el texto `ab ab ab ba`. La pretokenización produce
los fragmentos `ab`, ` ab`, ` ab` y ` ba`. ¿Qué par se fusiona primero, y cuántas veces aparece?

<details><summary>Respuesta</summary>

`a` + `b`, que aparece **3 veces** (una en `ab` y dos en ` ab`). El par espacio + `a` aparece 2 veces;
espacio + `b` y `b` + `a`, una vez cada uno. BPE fusiona primero el par más frecuente.

</details>

### P7

¿Por qué un tokenizador BPE a nivel de byte nunca necesita un *token* "desconocido"?

- A. Se entrena con todos los idiomas del mundo
- B. Sustituye las palabras raras por la palabra conocida más parecida
- C. Los 256 valores de byte están en el vocabulario base, así que se puede codificar cualquier texto UTF-8
- D. Descarta los caracteres que no reconoce

<details><summary>Respuesta</summary>

**C.** Cualquier texto es una secuencia de bytes, y cada byte es un *token*. Los caracteres raros
cuestan más *tokens* (una letra con tilde puede convertirse en dos *tokens* de byte), pero no se pierde
nada: `decode(encode(text)) == text`.

</details>

### P8

Encuentra el fallo en esta atención causal:

```python
scores = q @ k.T / math.sqrt(d)
weights = softmax(scores, axis=-1)
weights = np.where(causal_mask(t), weights, 0.0)
out = weights @ v
```

<details><summary>Respuesta</summary>

La máscara se aplica después del *softmax*. Cada fila se normalizó usando las puntuaciones del futuro,
así que las salidas anteriores siguen dependiendo de *tokens* futuros y las filas ya no suman 1.
Enmascara las **puntuaciones** con menos infinito antes del *softmax*:
`scores = np.where(causal_mask(t), scores, -np.inf)`.

</details>

### P9

Tu modelo tiene un vocabulario de 1.024 *tokens*. Antes de cualquier entrenamiento, ¿a qué valor
debería acercarse su pérdida del siguiente *token*, y qué sospecharías si empezara varias veces más
alta?

<details><summary>Respuesta</summary>

Alrededor de ln(1024) = **6.931** *nats*, la pérdida de una suposición uniforme (el lab mide 6.932 y
6.905). Una pérdida inicial mucho más alta significa que los *logits* iniciales están lejos de ser
iguales: pesos inicializados demasiado grandes, una normalización que falta o un error como objetivos
sin desplazar o una indexación incorrecta.

</details>

### P10

Usa la regla aproximada de Kaplan para estimar los parámetros sin *embeddings* de un modelo clásico de
solo decodificador con 8 capas y anchura 512 (anchura *feed-forward* 4 × 512).

<details><summary>Respuesta</summary>

12 × L × D² = 12 × 8 × 512² = **25.165.824**, unos 25 millones. La regla cuenta la atención (4D² por
capa) y la red *feed-forward* (8D² por capa) e ignora los sesgos y las normalizaciones; para el
`classic-small` del lab queda a menos de un 0,2 % del conteo exacto.

</details>

### P11

Un modelo de mezcla de expertos tiene 8 expertos de 100 millones de parámetros cada uno, envía cada
*token* a 2 de ellos y tiene 200 millones de parámetros fuera de los expertos. ¿Cuántos parámetros
totales y activos tiene?

<details><summary>Respuesta</summary>

Total = 200M + 8 × 100M = **1.000 millones**. Activos = 200M + 2 × 100M = **400 millones**. La memoria
tiene que albergar los 1.000 millones; el cómputo por *token* sigue a los 400 millones.

</details>

### P12 🔁 (lección 1.1)

Todas las páginas de las que aprendió un modelo eran correctas. Explica cómo puede aun así afirmar un
dato falso, usando el ejemplo del Ember T3 del lab.

<details><summary>Respuesta</summary>

El modelo produce continuaciones probables, no datos comprobados. Después de "on 2023 -", el
modelo de conteo había visto dos fechas con la misma frecuencia (la del Ember T3 y la del Corvid
Thermo X), y su contexto de tres palabras ya no contenía el nombre del producto, así que produjo
2023-05-22 en lugar de 2023-10-02. Cuando los datos importan, hacen falta fundamentación en documentos
y verificación.

</details>

### P13 🔁 (lección 1.1)

Con un presupuesto fijo de cómputo de entrenamiento, ¿qué recomendaba el artículo de Chinchilla?

- A. Entrenar el modelo más grande que quepa en memoria con los datos que haya disponibles
- B. Aumentar a la vez el tamaño del modelo y el número de *tokens* de entrenamiento
- C. Entrenar un modelo pequeño durante el mayor tiempo posible
- D. Gastar el presupuesto en post-entrenamiento

<details><summary>Respuesta</summary>

**B.** Para un entrenamiento óptimo en cómputo, los parámetros y los *tokens* de entrenamiento deben
escalarse en la misma proporción. Su modelo de 70.000 millones de parámetros, entrenado con más datos y
el mismo cómputo que Gopher, de 280.000 millones de parámetros, lo superó.

</details>

### P14

Dos modelos candidatos obtienen la misma puntuación en un *benchmark* público. Rio pregunta cuál usar
para el chatbot de soporte de Hearthwise. ¿Qué haces?

<details><summary>Respuesta</summary>

Trata la igualdad de puntuaciones como una razón para probar los dos, no como un desempate. Rellena la
hoja de comparación: parámetros totales y activos y memoria, longitud de contexto y caché KV, *tokens*
por palabra en tickets reales en inglés y en español, licencia y restricciones de residencia de los
datos. Después ejecuta los dos con el conjunto de evaluación propio de Hearthwise (módulo 02), y
revisa si el *benchmark* puede estar contaminado o usar otra configuración de *prompt*.

</details>

### P15

La atención de consultas agrupadas (GQA), tal como la usan las familias Qwen2.5 y Gemma 3, reduce
sobre todo:

- A. Los parámetros de las capas *feed-forward*
- B. La memoria de la caché KV durante la generación
- C. El tamaño del vocabulario
- D. El número de *tokens* de entrenamiento necesarios

<details><summary>Respuesta</summary>

**B.** Varias cabezas de consulta comparten una cabeza de clave y valor, así que se guardan en caché
menos claves y valores por *token*. En el lab, `modern-1b` con 4 cabezas de clave y valor para 16
cabezas de consulta necesita una cuarta parte de la caché que necesitaría sin compartir.

</details>

## Guía de puntuación

| Respuestas correctas | Qué significa |
|---|---|
| 13–15 | Tienes todo a punto para el módulo 02. |
| 10–12 | Bien. Vuelve a leer la autoevaluación de las lecciones de las preguntas que fallaste. |
| 7–9 | Repite los pasos del lab relacionados y los experimentos de Rómpelo antes de seguir. |
| 0–6 | Vuelve a recorrer las lecciones en orden; las salidas del lab son el mejor punto de apoyo. |

## ⏭️ Siguiente

[Módulo 02 — Del modelo base al asistente →](../02-decoding-post-training-evaluation/README.es.md)

---

[← Anterior: Lab](lab/README.es.md) · [🏠 Módulo 01](README.es.md) · [Siguiente: Chuleta →](cheatsheet.es.md)
