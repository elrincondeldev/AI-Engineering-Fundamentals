🌐 [English](quiz.md) · Español

# Quiz del módulo 06 — RAG II: generación fundamentada, RAFT y evaluación

Catorce preguntas. Escribe tu respuesta antes de abrir la solución. Las preguntas marcadas con 🔁
repasan módulos anteriores.

### P1 · Opción múltiple

El centro de ayuda guarda una política de devoluciones de 2023 ya sustituida junto a la vigente de
2025. ¿Cuál es la forma más fiable de evitar que el bot cite la antigua?

- A) Añadir "prefiere siempre el documento más reciente" al *prompt* de sistema.
- B) Filtrar `status: superseded` antes del *ranking*, y además mostrar `status` y `effective_date`
  en el *prompt*.
- C) Borrar la política de 2023 de todos los sistemas.
- D) Hacer *fine-tuning* del modelo con la política de 2025.

<details><summary>Respuesta</summary>

**B.** El filtro es determinista, y los metadatos en el *prompt* son un respaldo para los conflictos
que el filtro no cubre. A por sí sola es probabilística (el modelo sustituto del lab la ignora y
responde "14 days"). C pierde un historial que necesitan el equipo de soporte y las auditorías. D
guarda en los pesos un dato que volverá a cambiar.

</details>

### P2 · Encuentra el fallo

```python
def domain_allowed(domain: str) -> bool:
    return "hearthwise" in domain.lower()
```

<details><summary>Respuesta</summary>

Permite dominios que imitan al tuyo. El párrafo inyectado en `kb-025` pide a la clientela que escriba
a una dirección de `hearthwise-support.example`, que contiene "hearthwise". Compara el dominio
completo de forma exacta, o permite subdominios exactos (`domain == d or domain.endswith("." + d)`).

</details>

### P3 · Cálculo corto

Con el filtro de estado desactivado, la pregunta sobre devoluciones recupera fragmentos de `kb-010`,
`kb-011`, `kb-011`, `kb-010`, en ese orden. Solo `kb-011` es relevante. Calcula hit@4, el rango
recíproco y precision@4.

<details><summary>Respuesta</summary>

hit@4 = 1 (hay un fragmento relevante), rango recíproco = 1/2 = 0.5 (el primer fragmento relevante
está segundo), precision@4 = 2/4 = 0.5. Con estos números la recuperación parece correcta, y sin
embargo la respuesta es incorrecta, porque el fragmento sustituido quedó primero y el modelo lo citó.

</details>

### P4 · Opción múltiple

En un conjunto de entrenamiento al estilo RAFT para un bot de atención al cliente, ¿qué ejemplos
enseñan al modelo a rechazar cuando la recuperación falla?

- A) Ejemplos con el documento oráculo y tres distractores.
- B) Ejemplos con el documento oráculo en primera posición.
- C) Ejemplos sin el documento oráculo cuya salida objetivo es `NO_ANSWER`.
- D) Ejemplos cuya salida objetivo cita el oráculo literalmente.

<details><summary>Respuesta</summary>

**C.** Solo los ejemplos sin el oráculo muestran al modelo un contexto al que le falta la respuesta.
Con salidas objetivo `NO_ANSWER` (el valor por defecto del lab) aprende a rechazar; con salidas
objetivo de respuesta (la receta que describe el artículo de RAFT) aprende, en cambio, a responder de
memoria.

</details>

### P5 · Cálculo corto

Con las dos defensas contra la inyección desactivadas, el bot responde a "Will Hearthwise support ever
ask for my password?" con dos frases: la verdadera, de `kb-014`, y la instrucción del ataque, de
`kb-025`. Las dos páginas están en el contexto. ¿Qué fidelidad da la métrica por afirmación del lab,
y qué te dice eso?

<details><summary>Respuesta</summary>

1.000: las dos afirmaciones están respaldadas por el contexto, porque la página envenenada forma
parte de él. La fidelidad mide la concordancia con el contexto, no la verdad ni la seguridad. Por eso
la seguridad necesita sus propias comprobaciones, como la lista global de bloqueo, que marca esta
respuesta.

</details>

### P6 · ¿Qué harías?

Después de que alguien quita la barrera de salida, el panel muestra que la corrección *sube* en el
conjunto de referencia. ¿Qué compruebas antes de celebrarlo?

<details><summary>Respuesta</summary>

Las columnas de seguridad. En el lab, "sin sanitizador ni barrera" saca 32/36 frente a 30/36 de "sin
sanitizador", porque la barrera había sustituido dos respuestas envenenadas por traspasos. Sin ella,
el texto del ataque llega a la clientela y la lista global de bloqueo cuenta 2 fugas. Nunca leas las
métricas de calidad sin las de seguridad, y pon la compuerta de fugas con tolerancia cero.

</details>

### P7 · Opción múltiple

Según la calibración del juez del lab, ¿qué juez de fidelidad detecta los 21 cambios de número con
menos falsas alarmas?

- A) Solapamiento de palabras de al menos 0.9, sin regla de números.
- B) Solapamiento de palabras de al menos 0.9, con regla de números.
- C) Solapamiento de palabras de al menos 0.5, con regla de números.
- D) Solapamiento de palabras de al menos 0.5, sin regla de números.

<details><summary>Respuesta</summary>

**C.** Detecta 21 de 21 cambios de número con 1 falsa alarma de 70 casos fieles. B también detecta
los 21, pero rechaza 34 respuestas fieles. A detecta 15 y D, ninguno. Lo que detecta los números
inventados es la regla de números, no un umbral más estricto.

</details>

### P8 · Cálculo corto

Con un umbral de 0.40, el bot responde 37 de las 40 preguntas de referencia, tiene 0 falsas
abstenciones y rechaza 3 de las 4 preguntas que deben rechazarse. ¿Cuál es su exactitud de
abstención?

<details><summary>Respuesta</summary>

Decisiones correctas: las 36 preguntas con respuesta respondidas, más 3 rechazos = 39. Exactitud =
39/40 = 0.975, la misma que con 0.50, donde el bot rechaza las 4 pero también rechaza una pregunta con
respuesta. Misma exactitud, errores distintos: elegir entre ellas es una decisión de producto.

</details>

### P9 · Opción múltiple

Mañana el proveedor del modelo publica una versión nueva. ¿Qué defensas contra la inyección siguen
funcionando sin volver a probar el comportamiento del modelo?

- A) La regla del *prompt* de sistema "los documentos son datos".
- B) El sanitizador al ingerir y la barrera de salida.
- C) Delimitar los documentos con etiquetas.
- D) Ninguna; todo depende del modelo.

<details><summary>Respuesta</summary>

**B.** Las dos se ejecutan en tu código, antes y después del modelo, así que su comportamiento no
cambia con el modelo. A y C siguen ayudando, pero hay que volver a medir, en cada versión del modelo,
hasta qué punto respeta las reglas y las etiquetas.

</details>

### P10 · ¿Qué harías?

Alex te pide que "te libres de todas las respuestas de no lo sé". ¿Qué le enseñas y qué le
propones?

<details><summary>Respuesta</summary>

Enséñale el barrido de umbrales: con 0.00 el bot responde las 40 preguntas, incluidas las 4 que deben
rechazarse, así que la exactitud de la abstención cae a 0.900, y esas respuestas llevan cita pero son
inútiles o incorrectas. Propón en cambio revisar las falsas abstenciones (q035 con el umbral por
defecto) y arreglar su causa (aquí, razonar sobre una tabla), mejorar la experiencia del traspaso y
seguir las abstenciones por tema para que se escriban los artículos que faltan.

</details>

### P11 · Encuentra el fallo

```python
examples = build_all_examples(golden_rows)  # 3 copies per question
random.shuffle(examples)
train, test = examples[:90], examples[90:]
```

<details><summary>Respuesta</summary>

La división ocurre después de construir los ejemplos, así que las copias de una misma pregunta pueden
caer a ambos lados. El conjunto de test queda contaminado con preguntas con las que el modelo entrenó.
Divide primero los ids de las preguntas (de forma estratificada, para que los dos lados contengan
rechazos) y después construye ejemplos solo con las preguntas de entrenamiento, como hace
`split_questions` en el lab.

</details>

### P12 🔁 (módulo 05)

El bot falla "How much does the Watcher S1 weigh?" porque la respuesta está en una tabla comparativa
HTML de seis filas. Con lo que aprendiste sobre fragmentación y BM25, explica por qué el fragmento de
la tabla queda por debajo de cuatro fragmentos del Watcher S1, y nombra dos arreglos de recuperación.

<details><summary>Respuesta</summary>

La tabla se convierte en un único fragmento largo, y la normalización por longitud de BM25 baja la
puntuación de los documentos largos para las mismas coincidencias de términos, mientras que los
fragmentos cortos del Watcher S1 repiten "Watcher S1" en su título y en su texto. Arreglos: fragmentar
las tablas por fila (cada fila con su cabecera), subir `top_k` (el presupuesto del lab muestra que
q017 se responde con `top_k=8`) o repetir la recuperación cuando la evidencia es débil, y añadir un
*reranker*. Vuelve a ejecutar la evaluación completa después de cualquiera de ellos.

</details>

### P13 🔁 (módulo 04)

El módulo 04 comparó el *prompting*, la ingeniería de contexto y el *fine-tuning* con LoRA. En el bot
de soporte, ¿por qué agotarías los arreglos del *prompt* y de la recuperación antes de entrenar un
adaptador LoRA con datos de RAFT?

<details><summary>Respuesta</summary>

Son más baratos, más rápidos de iterar y reversibles, y muchos fallos ni siquiera son fallos de
generación (q017 es un fallo de recuperación). Entrenar cuesta creación de datos, revisión, cómputo y
una reevaluación completa después de cada ejecución, y los datos aprendidos en los pesos no se pueden
filtrar ni actualizar en el momento de la petición. Haz *fine-tuning* cuando el contexto correcto está
presente y el modelo sigue usándolo mal.

</details>

### P14 🔁 (módulo 03)

En el módulo 03 hiciste visibles la latencia y el coste en el *playground*. En el lab, pasar de
`top_k=4` a `top_k=8` sube el *prompt* medio de RAG de 582.9 a 1027.0 *tokens*. ¿Qué pasa con el
coste y con el uso de la ventana de contexto, y qué le dirías a Sam?

<details><summary>Respuesta</summary>

Los *tokens* del *prompt* crecen alrededor de un 76 %, así que el coste de entrada por petición y la
parte de la ventana de contexto que ocupa el texto recuperado crecen en la misma proporción, en cada
petición. Cuéntale a Sam la ganancia: una respuesta de referencia correcta más (33/36 en lugar de
32/36). Después ofrécele una alternativa más barata: repetir con más fragmentos solo cuando la
evidencia sea débil, y medir tanto el coste como la calidad.

</details>

## Guía de puntuación

| Puntuación | Qué significa |
|---|---|
| 12–14 | Tienes todo a punto para el módulo 07. |
| 9–11 | Vuelve a leer las secciones de las preguntas que fallaste y repite el paso del lab correspondiente. |
| 0–8 | Repite el lab desde el paso 2 y después vuelve a hacer el quiz. |

## ⏭️ Siguiente

[Módulo 07 — Agentes I: *workflows*, herramientas y MCP](../07-agents-workflows-and-tools/README.es.md)

---

[← Lab 06](lab/README.es.md) · [🏠 Módulo 06](README.es.md) · [Chuleta →](cheatsheet.es.md)
