🌐 [English](quiz.md) · Español

# Quiz del módulo 08 — El agente Ask-the-Web

13 preguntas, unos 20 minutos. Responde primero y después abre la explicación. Las preguntas marcadas
con 🔁 repasan módulos anteriores.

### P1 — Elegir la autonomía

Hearthwise quiere una tarea diaria que descargue la página de la política de devoluciones y publique un
resumen de tres líneas en un canal. ¿Qué diseño encaja mejor?

- A. Un agente ReAct con `search_web` y `open_page`
- B. Un *workflow* fijo: obtener la página, resumirla, publicarla
- C. Un supervisor con un investigador y un verificador de hechos
- D. Una búsqueda en árbol de Monte Carlo sobre los resúmenes posibles

<details><summary>Respuesta</summary>

**B.** Los pasos se conocen de antemano y nada depende de resultados intermedios. Cada nivel extra de
autonomía (A, C, D) añade coste, modos de fallo y trabajo de evaluación sin añadir capacidad.

</details>

### P2 — Encuentra el fallo

```python
for _ in range(max_steps):
    text = provider.complete(messages).text
    kind, arg = parse_action(text)
    if kind == "final":
        return text
    obs = tools.call(kind, arg)
    messages.append(Message("assistant", text + "\nObservation: " + obs))
```

¿Qué está mal, y qué puede hacer un modelo real por culpa de ello?

<details><summary>Respuesta</summary>

Hay dos problemas. La observación se escribe dentro del mensaje del *asistente*, así que el modelo no
puede distinguir lo que dijo él de lo que devolvió la herramienta; debería ser un mensaje aparte del
arnés. Y no hay `stop=["\nObservation:"]`, así que un modelo real puede escribir su propia observación
falsa y continuar a partir de ella. (Además, no pasa nada cuando se agota el presupuesto: el bucle
simplemente termina sin respuesta).

</details>

### P3 — Coste del bloc de notas (cálculo)

Una ejecución ReAct hace cuatro llamadas al modelo con *prompts* de 300, 600, 900 y 1.200 *tokens*.
¿Cuántos *tokens* de *prompt* usó, y cuántos añadiría un quinto paso al mismo ritmo de crecimiento?

<details><summary>Respuesta</summary>

300 + 600 + 900 + 1.200 = **3.000** *tokens* de *prompt*. Una quinta llamada llevaría unos 1.500
*tokens*, así que el total subiría a unos 4.500: cada paso añade más que el anterior porque se reenvía la
transcripción completa.

</details>

### P4 — ReWOO

En las 24 tareas del lab con todas las protecciones activadas, ReWOO iguala el éxito de ReAct con 2,00
llamadas al modelo por tarea en lugar de 4,25. ¿Qué afirmación explica dónde *pierde* ReWOO?

- A. Necesita más llamadas al modelo en las preguntas de varios saltos
- B. No puede adaptar su plan a lo que contienen las observaciones
- C. No puede citar fuentes
- D. Siempre responde "unknown" en las tareas sin respuesta

<details><summary>Respuesta</summary>

**B.** El planificador escribe todas las llamadas a herramientas antes de ver ninguna evidencia, así que
lee dos resultados incluso cuando el primero es decisivo (3,88 llamadas a herramientas por tarea frente a
3,25) y no puede reformular una consulta que no devolvió nada útil. A es falsa: siempre usa dos llamadas.
C y D no son específicas de ReWOO.

</details>

### P5 — El evaluador de Reflexion

Alguien de tu equipo propone este evaluador para Reflexion: "reintenta si la respuesta no coincide con la
respuesta de referencia de tasks.jsonl". ¿Qué le dices?

<details><summary>Respuesta</summary>

Que filtra la respuesta de referencia al agente. La evaluación parecería excelente y no diría nada sobre
producción, donde no existe ninguna referencia. El evaluador solo debe usar señales disponibles en tiempo
de ejecución: señales de *spam* en una página citada, texto inyectado que se repite en la respuesta, una
fuente más reciente que no coincide, una respuesta "unknown".

</details>

### P6 — Aritmética de la búsqueda en árbol (cálculo)

Una búsqueda en haz tiene 2 niveles, 3 páginas candidatas por nivel y una anchura de haz de 2. ¿Cuántas
llamadas al modelo de valor hace, suponiendo que cada candidata produce un nodo?

<details><summary>Respuesta</summary>

Nivel 1: 3 hijos de la raíz. Nivel 2: 2 nodos conservados × 3 candidatas = 6. Total: **9** llamadas de
valor. El coste crece con el factor de ramificación por la anchura del haz en cada nivel, y por eso el lab
agrupa en un solo nivel los datos que comparten una página.

</details>

### P7 — Una función de valor engañada

Con todas las protecciones desactivadas, la búsqueda en haz del lab responde 200 a "How many devices can a
single HomeMesh 2.0 network support?" (la correcta: 250). ¿Cuál es la mejor solución?

- A. Ampliar la búsqueda a 5 candidatas
- B. Eliminar las páginas duplicadas antes de calcular la coincidencia, o mostrar los niveles de
  credibilidad al modelo de valor
- C. Cambiar a la búsqueda en árbol de Monte Carlo (MCTS)
- D. Bajar la temperatura

<details><summary>Respuesta</summary>

**B.** El mensaje del foro y su copia con `?utm_source=` "coinciden" en 200 y reciben la bonificación por
coincidencia. A lo empeora (21/24 en lugar de 22/24 en el lab), C usa la misma función de valor y D no
cambia nada en un modelo de valor determinista.

</details>

### P8 — Radio de impacto multiagente

El agente investigador lee una página con una instrucción inyectada y la repite en su resumen en texto
libre. ¿Qué cambio de diseño impide de forma más directa que la instrucción llegue a la respuesta final?

<details><summary>Respuesta</summary>

Hacer estructurado el traspaso: el supervisor solo acepta del investigador campos tipados (valor,
dirección web, fecha), nunca su prosa. En el lab, el traspaso en texto libre dejó que el anuncio llegara
a quien usa el asistente; el traspaso estructurado lo bloqueó incluso con el sanitizador desactivado. El
mínimo privilegio para el agente que lee la web limita aún más el daño.

</details>

### P9 — ¿A2A o MCP?

Asocia cada necesidad con el protocolo Agent2Agent (A2A) o con el Model Context Protocol (MCP): (1)
exponer `search_web` a cualquier *framework* de agentes; (2) dejar que el agente de precios de una
empresa socia, cuyo código no puedes ver, acepte una tarea de "comparar precios" e informe más tarde.

<details><summary>Respuesta</summary>

(1) **MCP**: conecta agentes con herramientas y datos. (2) **A2A**: al otro lado hay un agente opaco que
trabaja en una tarea con su propio ciclo de vida (*submitted*, *working*, *input-required*,
*completed*...) y devuelve artefactos; lo descubres a través de su tarjeta de agente.

</details>

### P10 — Leer una ablación

Con todas las protecciones activadas, ReAct tiene éxito en 24/24. Desactivar solo la credibilidad da
21/24; desactivar todas las protecciones da 22/24. ¿Cómo puede ser que quitar *más* protecciones haga
fallar *menos* tareas?

<details><summary>Respuesta</summary>

Las protecciones interactúan. Con la credibilidad desactivada pero la recencia activada, "prefiere la
fuente más reciente" elige un mensaje del foro más nuevo y una página de *spam* más nueva por encima de la
especificación y del informe de MarketLens. Con la recencia también desactivada, los conflictos se
resuelven contando las fuentes que coinciden y, después, por "la primera que se vio", lo que casualmente
favorece a las mejores páginas en dos de esas tareas (task02 y task21). Haz la ablación de las
protecciones de una en una *y* en combinación.

</details>

### P11 🔁 (módulo 06)

En el módulo 06, el chatbot de soporte tenía que negarse a responder cuando el centro de ayuda no
contenía la respuesta. ¿Qué dos métricas de este módulo cumplen el mismo papel para el agente
Ask-the-Web?

<details><summary>Respuesta</summary>

El acierto en "unknown" en las tareas sin respuesta (¿se abstuvo cuando debía?) y el recuento de falsos
"unknown" en las tareas con respuesta (¿se abstuvo cuando no debía?). Informar solo de la primera premia a
un agente que responde "unknown" a todo.

</details>

### P12 🔁 (módulo 05)

La búsqueda del agente usa BM25 (*Best Matching 25*). La consulta "smart thermostats sold spain 2025"
pone primero una página de *spam*
rellena de palabras clave cuando la reordenación está desactivada. ¿Por qué la favorece BM25, y qué idea
del módulo 05 limita el efecto?

<details><summary>Respuesta</summary>

BM25 premia los términos de la consulta que aparecen a menudo, y el relleno de palabras clave los repite.
La saturación de la frecuencia de términos (el parámetro `k1`) limita cuánto añade cada repetición, pero
no puede distinguir una página honesta de una rellena. Por eso el lab añade la reordenación por
credibilidad y las señales de *spam* por encima de la recuperación.

</details>

### P13 🔁 (módulo 07)

En el módulo 07 expusiste herramientas mediante MCP y revisaste sus permisos con Noor. El agente
Ask-the-Web solo tiene `search_web` y `open_page`. ¿Por qué importa eso frente a la inyección indirecta de
*prompts*?

<details><summary>Respuesta</summary>

El mínimo privilegio limita el radio de impacto. Una instrucción inyectada como mucho puede cambiar lo
que el agente *dice*; no puede enviar correos, comprar nada ni cambiar datos, porque no existe ninguna
herramienta así. Por eso también el riesgo que queda (una respuesta contaminada) se pone a prueba con
`must_not_contain` y con las comprobaciones de dominios no citados.

</details>

## 📊 Guía de puntuación

| Aciertos | Qué significa |
|---|---|
| 12–13 | Tienes todo a punto para el módulo 09. |
| 9–11 | Buen nivel; vuelve a leer las autoevaluaciones de las lecciones donde fallaste preguntas. |
| 6–8 | Repite los pasos del lab de esas lecciones y rehaz sus secciones "Pruébalo". |
| 0–5 | Vuelve a trabajar las lecciones con el lab abierto al lado. |

## ⏭️ Siguiente

[Módulo 09 — Modelos de razonamiento →](../09-reasoning-models/README.es.md)
