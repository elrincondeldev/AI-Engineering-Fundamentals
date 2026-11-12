🌐 [English](quiz.md) · Español

# Quiz del módulo 04 — Adaptar LLM

Responde primero cada pregunta en papel y después abre la respuesta. Las preguntas marcadas con 🔁
repasan módulos anteriores. Todos los números salen del lab del módulo 04.

### P1

Un enrutador va a funcionar con los tickets del mes que viene. ¿Qué diseño de evaluación da la
estimación más honesta de su exactitud?

- A. Una partición aleatoria 80/20 de los 1.224 tickets.
- B. Una partición aleatoria 80/20 después de quitar las 24 copias reenviadas.
- C. Entrenar con los tickets anteriores al 2025-09-01 y probar con los tickets desde esa fecha,
  después de quitar las copias reenviadas.
- D. Evaluar con los mismos tickets que se usaron para elegir los ejemplos *few-shot*, para
  aprovechar al máximo los datos.

<details><summary>Respuesta</summary>

**C.** Imita la producción: todo lo que usaste para construir viene del pasado y la prueba viene del
futuro, deriva incluida. En el lab, el mismo *prompt* obtuvo 0.959 con A, 0.921 con B y 0.828 con C.
B quita los peores duplicados, pero sigue mezclando pasado y futuro. D mide memoria, no enrutamiento.

</details>

### P2

Encuentra el fallo en este constructor de *prompts*.

```python
def user_message(ticket: Ticket) -> str:
    context = (f"language={ticket.language}; product={ticket.product}; "
               f"resolution={ticket.resolution_code}")
    return f"Customer context: {context}\nTicket: {clean_text(ticket.body)}"
```

<details><summary>Respuesta</summary>

`resolution_code` lo escribe alguien del equipo de soporte después de resolver el ticket y corresponde
uno a uno con la intención. No existe cuando llega un ticket nuevo, así que ponerlo en el *prompt* es
una fuga del objetivo (*target leakage*). En el lab subió la exactitud de 0.828 a 0.920, una mejora
que desaparecería con el primer ticket real. Construye el contexto a partir de una lista de campos
permitidos que se conocen en el momento de predecir.

</details>

### P3

En el lab, con definiciones de etiquetas bilingües, ¿qué afirmación sobre los ejemplos *few-shot* es
verdadera?

- A. Ocho ejemplos aleatorios ganan a ocho ejemplos parecidos porque cubren más intenciones.
- B. Ocho ejemplos parecidos obtuvieron 0.828 frente a 0.728 con ocho aleatorios.
- C. Los ejemplos nunca ayudan cuando las definiciones de las etiquetas son buenas.
- D. Un ejemplo parecido siempre es tan bueno como ocho.

<details><summary>Respuesta</summary>

**B.** Los ejemplos parecidos comparten palabras con el ticket y le dan al aprendiz de juguete
pruebas reales; los aleatorios rara vez lo hacen. A es falsa en este lab (los ejemplos equilibrados,
uno por intención, solo llegaron a 0.740). C la contradice el salto de 0.723 a 0.828. D también es
falsa: un ejemplo parecido obtuvo 0.782.

</details>

### P4

Alguien del equipo añade "Think step by step" al *prompt* de enrutamiento. En el conjunto de prueba,
la exactitud se queda en 0.828 y los *tokens* de respuesta medios pasan de 9.4 a 35.7. ¿Qué haces?

<details><summary>Respuesta</summary>

Quitarlo para esta tarea o, como mínimo, no lanzarlo sin una mejora en un modelo real. La cadena de
pensamiento ayuda en problemas de varios pasos; enrutar es una tarea de reconocimiento de un solo
paso. El cambio multiplicó los *tokens* de salida, que suelen ser la parte más lenta y más cara de una
petición, sin ninguna mejora de exactitud. Conserva el análisis de "Answer:" por si más adelante
necesitas el *prompting* paso a paso para una tarea de varios pasos.

</details>

### P5

Encuentra el fallo.

```python
def system_message(request_id: str) -> Message:
    header = f"Request {request_id} received at {datetime.now().isoformat()}\n"
    return Message("system", header + ROUTER_INSTRUCTIONS)
```

<details><summary>Respuesta</summary>

La primera línea cambia en cada petición, así que no hay dos *prompts* que compartan prefijo y la
caché de prefijo nunca acierta. En la simulación del lab, el *prompt* de sistema estable tuvo 399
aciertos de 400 y se sirvió desde la caché una proporción de 0.496 de los *tokens* del *prompt*; con un
ID de petición y una fecha al principio tuvo 0 aciertos. Pon los datos variables al final del contexto
o en los metadatos de la petición.

</details>

### P6

Tienes ocho ejemplos ordenados por similitud. ¿Dónde debería ir el más parecido?

- A. Primero, justo después del mensaje de sistema.
- B. En el medio, para que esté rodeado de contexto.
- C. Al final, justo antes del ticket.
- D. En cualquier sitio: los modelos ignoran la posición.

<details><summary>Respuesta</summary>

**C.** Estudios como "Lost in the Middle" vieron que los modelos usan la información del principio o
del final de un contexto largo de forma más fiable que la del medio. El ensamblador del lab pone
primero el *prompt* de sistema estable y el ejemplo más relevante junto al ticket. D es falsa para
muchos modelos reales, aunque el juguete del lab no tiene en cuenta el orden por defecto; por eso la
posición se prueba con una sonda.

</details>

### P7

Con un presupuesto de 700 *tokens*, quitar los ejemplos casi duplicados ahorra 102,1 *tokens* por
ticket, pero la exactitud baja de 0.843 a 0.830. Tu responsable pregunta si conviene mantener la
deduplicación. ¿Qué respondes?

<details><summary>Respuesta</summary>

Es un compromiso que se decide con números, no una regla. La deduplicación recortó el coste en
alrededor de un 15 % del *prompt* (583.0 frente a 685.1 *tokens*) y perdió 1,3 puntos, porque los
ejemplos duplicados actuaban como votos extra para la etiqueta correcta. Repite la comparación con el
modelo real y después pondera el ahorro de *tokens* a tu volumen frente a los tickets mal enrutados de
más. Una opción intermedia es mantener el presupuesto y deduplicar solo las copias exactas.

</details>

### P8

Calcula el tamaño de LoRA para una matriz de pesos con 4.096 salidas y 1.024 entradas con rango 16.
¿Qué proporción de la matriz completa supone?

<details><summary>Respuesta</summary>

`r · (d_in + d_out) = 16 · (1,024 + 4,096) = 81,920` números entrenables. La matriz completa tiene
`1,024 · 4,096 = 4,194,304`, así que LoRA entrena una proporción de 0.01953, alrededor del 2 %.
Compruébalo con `uv run lab lora-params --d-in 1024 --d-out 4096`.

</details>

### P9

Un modelo tiene 13.000 millones de parámetros. ¿Cuánta memoria necesitan solo sus pesos con
precisión de 16 bits y de 4 bits, y qué técnica entrena adaptadores encima de la versión de 4 bits?

<details><summary>Respuesta</summary>

16 bits: `13 × 10⁹ × 2 bytes = 26.0 GB`. 4 bits: `13 × 10⁹ × 0.5 bytes = 6.5 GB`. QLoRA guarda la base
congelada en 4 bits y entrena encima adaptadores LoRA de 16 bits. El entrenamiento necesita memoria
extra para las activaciones y para los gradientes y el estado del optimizador de los adaptadores, así
que estos números son un límite inferior.

</details>

### P10

¿Por qué LoRA inicializa `B` a cero?

- A. Para que el adaptador ocupe menos en disco.
- B. Para que el modelo adaptado empiece siendo exactamente igual al modelo preentrenado.
- C. Para que la actualización sea de rango completo.
- D. Porque A y B siempre deben ser iguales.

<details><summary>Respuesta</summary>

**B.** Con `B = 0`, la actualización `(alpha/r)·B·A` es cero, así que el entrenamiento parte del
comportamiento del modelo base y solo se aleja de él tanto como justifiquen los datos. A es aleatoria
para que el gradiente de B no sea cero. C y D son falsas: la actualización tiene como mucho rango r, y
A y B tienen formas distintas.

</details>

### P11

Tras 4.000 pasos de entrenamiento con los tickets de septiembre, un enrutador con *fine-tuning*
completo obtiene 0.855 en octubre–diciembre y 0.792 en agosto. La base congelada obtenía 0.828 y
0.917. Alguien del equipo quiere lanzarlo porque "los tickets nuevos mejoraron". ¿Qué haces?

<details><summary>Respuesta</summary>

No lanzarlo. El modelo olvidó cómo enrutar los tickets antiguos (olvido catastrófico) y sobreajustó
los 97 tickets de septiembre: con solo 100 pasos obtenía 0.927 en los nuevos y 0.906 en los antiguos.
Usa el conjunto de datos antiguos como compuerta de lanzamiento, entrena menos (o para antes) y mezcla
tickets antiguos en el entrenamiento: la repetición de 100 tickets antiguos mantuvo agosto en 0.865
incluso con 4.000 pasos. No des por hecho que LoRA lo resuelve; LoRA de rango 4 también cayó, a
0.812.

</details>

### P12 🔁 (módulo 03)

La conversación del *playground* ha crecido más allá de la ventana de contexto del modelo. ¿Cuál es
la mejor primera estrategia?

- A. Quitar el *prompt* de sistema, porque el modelo ya lo ha visto.
- B. Conservar el *prompt* de sistema y los últimos turnos literalmente, y sustituir los turnos
  antiguos por un resumen.
- C. Enviarlo todo y dejar que el proveedor trunque.
- D. Empezar en silencio una conversación nueva.

<details><summary>Respuesta</summary>

**B.** El modelo no tiene estado; tu sistema decide qué ve en cada turno. Conserva lo obligatorio (el
*prompt* de sistema), lo más relevante (los turnos recientes) y una versión compacta del resto. La
lección 4.3 lo generaliza en un ensamblador con presupuesto, prioridades e informe de descartes. A
pierde las reglas, C corta por posición en lugar de por importancia y D sorprende a quien está usando
el chat.

</details>

### P13 🔁 (módulo 02)

¿Qué ajuste de decodificación encaja con un enrutador cuya salida debe ser una de ocho etiquetas?

- A. Temperatura 1.0, para tener respuestas más diversas.
- B. Temperatura 0 (decodificación voraz), para tener respuestas repetibles.
- C. Un *top-p* alto, para que las etiquetas raras tengan una oportunidad.
- D. Un `max_tokens` grande, para que el modelo pueda explicarse.

<details><summary>Respuesta</summary>

**B.** Enrutar necesita la etiqueta más probable cada vez, y el mismo ticket debería recibir siempre
la misma ruta. La función `classify` del lab llama al modelo con `temperature=0.0` y un `max_tokens`
pequeño. A y C añaden aleatoriedad que solo genera rutas incoherentes; D invita a salidas largas que
cuestan *tokens* y complican el análisis.

</details>

### P14 🔁 (módulo 02)

¿Qué relación tiene LoRA con el ajuste fino supervisado (SFT, *supervised fine-tuning*)?

- A. LoRA elimina la necesidad de ejemplos etiquetados.
- B. LoRA es una forma de hacer SFT (u otro entrenamiento) actualizando solo pequeños adaptadores de
  bajo rango.
- C. LoRA es un método de decodificación que se usa después del SFT.
- D. LoRA es un modelo de recompensa para la optimización de preferencias.

<details><summary>Respuesta</summary>

**B.** El SFT define *con qué* entrenas (pares de entrada y salida deseada, con la pérdida sobre la
respuesta); LoRA define *qué* parámetros cambian (una actualización de bajo rango en lugar de todos
los pesos). Sigues necesitando buenos datos etiquetados, y los problemas de calidad de los datos, como
los duplicados o las etiquetas erróneas, perjudican a LoRA exactamente igual.

</details>

## Guía de puntuación

Date un punto por pregunta.

| Puntuación | Qué significa | Qué hacer |
|---|---|---|
| 12–14 | Sabes aplicar las ideas del módulo a casos nuevos | Pasa al módulo 05 |
| 9–11 | Base sólida, con lagunas | Relee la sección de la lección de cada fallo y repite el paso del lab correspondiente |
| 0–8 | Las ideas todavía no se han asentado | Repite los pasos 2, 4, 6 y 9 del lab y vuelve a hacer el quiz dentro de unos días |

## ⏭️ Siguiente

[Módulo 05 — RAG I: recuperación →](../05-rag-retrieval/README.es.md)
