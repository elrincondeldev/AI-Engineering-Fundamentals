🌐 [English](quiz.md) · Español

# Quiz del módulo 10 — *Deep research* y despliegue local

14 preguntas. Responde cada una antes de abrir la solución. Las preguntas marcadas con 🔁 repasan
módulos anteriores.

### P1

¿Qué paso de un *pipeline* de *deep research* debería ser código simple en lugar de una llamada al
modelo?

- A. Planificar las subconsultas
- B. Redactar el informe final
- C. Comprobar que cada número citado aparece en la página citada
- D. Extraer frases de evidencia de una página

<details><summary>Respuesta</summary>

**C.** La comprobación de las citas sigue una regla exacta y el texto que comprueba no debe poder
convencerla. Planificar y redactar requieren juicio (encaja un modelo de razonamiento), y la extracción
es una tarea acotada para un modelo cuya salida verificas después en código.

</details>

### P2

La traza de un informe dice: 1 llamada de planificación, 20 páginas abiertas por el lector, 1 llamada
de redacción. ¿Cuántas llamadas al modelo costó el informe y qué proporción se fue en leer? ¿Dónde
mirarías primero para recortar el coste?

<details><summary>Respuesta</summary>

22 llamadas, 20 de ellas para leer: en torno al 91 %. Recorta el coste donde están las llamadas: usa un
modelo más pequeño para el lector, abre menos páginas por búsqueda, descarta páginas antes con
comprobaciones baratas (credibilidad, duplicados) y guarda en caché las páginas que abren varias
subconsultas.

</details>

### P3 — Encuentra el fallo

```python
quotes = [q.strip() for q in out.text.splitlines() if q.strip() and q.strip() != "NONE"]
words = " ".join(quotes).split()
passage = " ".join(words[:max_words])
```

<details><summary>Respuesta</summary>

Las citas nunca se verifican contra la página. Un modelo que parafrasea o inventa una frase ("The
Ember T3 costs €99") crearía evidencia que ninguna fuente contiene. El lab solo conserva
`[q for q in quotes if q in text]` y cuenta las rechazadas, lo que además hace visible en las
estadísticas un modelo que se porta mal.

</details>

### P4 — ¿Qué harías?

Alguien de tu equipo te enseña la tabla de ablaciones: quitar solo el sanitizador no cambia nada (17 de
17 hechos, 0 trampas, 0 PII). Propone borrar el sanitizador para simplificar el código. ¿Qué le
respondes?

<details><summary>Respuesta</summary>

Mantenlo. No cambia nada solo porque el filtro de credibilidad descarta el foro antes de que el lector
lo abra. Con las dos defensas desactivadas, la instrucción inyectada y los datos personales de la
persona usuaria del foro llegan al informe (2 trampas, 2 filtraciones de PII). Defensa en profundidad
significa que cada capa se prueba por separado y en combinación, y se mantiene porque las demás pueden
fallar.

</details>

### P5

El mismo medio midió el Corvid Thermo X en 0,9 W en 2024 y en 0,4 W en 2025, después de una
actualización de *firmware*. ¿Qué debería hacer el informe?

- A. Promediarlas: unos 0,65 W
- B. Citar las dos sin comentarios
- C. Usar la cifra de 2025 e indicar que sustituye a la de 2024
- D. Usar la cifra de 2024 porque se publicó primero

<details><summary>Respuesta</summary>

**C.** La página más nueva sustituye explícitamente a la antigua. Promediar inventa un número que nadie
midió, y citar las dos sin comentarios deja que quien lee elija la equivocada. La ponderación por
actualidad por sí sola no elimina la página antigua; la regla de sustitución (mismo medio, mismo
producto, quedarse con la más reciente) sí.

</details>

### P6 — Cálculo

¿Cuántos GB necesitan los pesos de un modelo de 14.000 millones de parámetros a 4 bits con una escala
de 16 bits por cada grupo de 32 pesos? (1 GB = 10^9 bytes).

<details><summary>Respuesta</summary>

4 + 16/32 = 4,5 bits por peso. 14 x 10^9 x 4,5 / 8 = 7,875 x 10^9 bytes, unos **7,88 GB**, antes de
la caché de claves y valores y del margen adicional.

</details>

### P7 — Cálculo

Un modelo tiene 32 capas, 8 cabezas de claves y valores y una dimensión de cabeza de 128, y guarda la
caché en 16 bits. ¿Qué tamaño tiene la caché de claves y valores para 4 conversaciones de 4.096
*tokens* cada una?

<details><summary>Respuesta</summary>

Por *token*: 2 x 32 x 8 x 128 x 2 bytes = 131.072 bytes (128 KiB). Por 4.096 *tokens* y por 4
conversaciones: 2.147.483.648 bytes, unos **2,15 GB**. La caché crece linealmente tanto con el
contexto como con el lote.

</details>

### P8

En el lab, la cuantización int4 con una sola escala para todo el tensor convierte en cero el 99,6 % de
los pesos. ¿Por qué?

- A. 4 bits no pueden representar números negativos
- B. Unos pocos valores atípicos grandes fijan la escala, así que los escalones son demasiado gruesos para los pesos pequeños normales
- C. El redondeo siempre va hacia cero
- D. Las escalas también se guardan en 4 bits

<details><summary>Respuesta</summary>

**B.** La escala es max|w| / 7. Con valores atípicos diez veces mayores que los pesos normales, la
mayoría de los pesos son menores que medio escalón y se redondean a 0. Las escalas por canal y por
grupo dan a cada bloque su propio rango, y por eso los grupos de 32 mantienen el error de salida cerca
de 0,11 en lugar de 0,92.

</details>

### P9

En el simulador, el *batching* estático da un p50 del tiempo hasta el primer *token* de unos 10
segundos y el *batching* continuo, de unos 98 ms, con las mismas peticiones y los mismos costes por
paso. ¿Qué explica la mayor parte de la diferencia?

<details><summary>Respuesta</summary>

Las colas. Con *batching* estático, una petición nueva espera hasta que termina la respuesta más larga
del lote actual; con *batching* continuo, se une en el siguiente paso de decodificación. El tiempo por
*token* de salida es parecido en los dos (36,8 ms y 34,1 ms); lo que cambia es la espera.

</details>

### P10 — Cálculo

Decodificación especulativa con una tasa de aceptación de 0,8, 4 *tokens* de borrador por ronda y un
paso del borrador que cuesta el 5 % de un paso del objetivo: ¿cuántos *tokens* produce de media cada
pasada del objetivo y cuál es la aceleración esperada? Usa *tokens* = (1 - α^(k+1)) / (1 - α) y
aceleración = *tokens* / (k x c + 1).

<details><summary>Respuesta</summary>

*tokens* = (1 - 0,8^5) / 0,2 = 3,362; aceleración = 3,362 / (4 x 0,05 + 1) = **2,80x**. La
distribución de salida no cambia, porque el modelo objetivo comprueba cada *token* del borrador.

</details>

### P11 — ¿Qué harías?

El modelo de costes muestra 1,80 € al día para una API y 2,54 € al día para una máquina local con el
volumen actual. Jordan dice: "Entonces usamos la API". ¿Qué respondes?

<details><summary>Respuesta</summary>

En este proyecto, no: la regla de Noor dice que los *prompts* y los datos de mercado no deben salir de
la máquina, así que la API no es una opción a ningún precio. El modelo de costes le dice a Jordan
cuánto cuesta la privacidad (aquí 0,74 € al día, o solo los 0,23 € al día de energía si el portátil ya
existe) y cuándo lo local pasa a ser más barato de todos modos (1,4 veces el volumen actual). Los
precios son valores de ejemplo; vuelve a calcularlo con los reales.

</details>

### P12 🔁 (módulo 05)

La búsqueda del lab es Okapi BM25. ¿Con qué pregunta es más probable que BM25 no encuentre la página
correcta en la copia de la web?

- A. "smart thermostat sales 2025 growth Mexico" (el informe usa estas palabras)
- B. Una pregunta en inglés cuya respuesta solo aparece en una página en español
- C. "Corvid Thermo X standby power"
- D. "household electricity price kWh Portugal"

<details><summary>Respuesta</summary>

**B.** BM25 puntúa los términos compartidos. Una página en español casi no comparte palabras con una
consulta en inglés, así que queda abajo en el ranking aunque contenga la respuesta. Ahí ayudan los
*embeddings* densos multilingües o traducir la consulta.

</details>

### P13 🔁 (módulo 08)

El planificador del lab escribe todas las subconsultas antes de cualquier búsqueda y después reflexiona
tras cada ronda. ¿Qué idea anterior se parece más a "primero planificar, después ejecutar"?

- A. ReAct: intercalar un pensamiento y una acción en cada paso
- B. ReWOO: planificar las llamadas a herramientas por adelantado, después ejecutarlas y combinar los resultados
- C. Búsqueda en árbol sobre muchas secuencias de acciones alternativas
- D. Votación por autoconsistencia sobre respuestas muestreadas

<details><summary>Respuesta</summary>

**B.** Planificar por adelantado ahorra llamadas al modelo y hace predecible el coste. La ronda de
reflexión añade un poco de la adaptabilidad de ReAct: si falta evidencia, el bucle lanza búsquedas de
seguimiento.

</details>

### P14 🔁 (módulo 09)

El módulo 09 mostró que el cómputo extra en tiempo de inferencia (muestrear varias respuestas,
verificadores) compensa en problemas con varios pasos dependientes. ¿En qué parte del *pipeline* de
investigación lo gastarías?

<details><summary>Respuesta</summary>

En la planificación y la síntesis: una llamada para cada una por informe, donde un error afecta a todo lo que
viene después (un aspecto que falta, una comparación equivocada). No en la lectura, que se llama una
vez por página y que de todos modos verifica el código. Un verificador para la síntesis puede ser
código barato: la comprobación de citas del evaluador es uno.

</details>

## Guía de puntuación

| Puntuación | Qué significa |
|---|---|
| 12–14 | Todo a punto para el proyecto final. |
| 9–11 | Vuelve a leer las lecciones de las preguntas que fallaste y repite el paso del lab correspondiente. |
| 0–8 | Vuelve a trabajar el lab, sobre todo las ablaciones (paso 5) y el estimador de memoria (paso 6). |

## ⏭️ Siguiente

Has terminado el último módulo. Reúne todo lo aprendido en el
[proyecto final (*capstone*): la revisión de lanzamiento de Hearthwise Assist](../../capstone/README.es.md).
