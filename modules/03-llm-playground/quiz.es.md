🌐 [English](quiz.md) · Español

# Quiz del módulo 03 — Construye un playground de LLM

Trece preguntas. Responde cada una en papel antes de abrir la respuesta. Las preguntas marcadas con 🔁
repasan módulos anteriores.

### P1 — ¿Qué componente cumple la promesa?

Noor exige que ningún dato personal de la clientela llegue a una API (interfaz de programación de
aplicaciones, *application programming interface*) externa. ¿Dónde debe imponerlo el *playground*?

- A. En el cliente web, ocultando los modelos externos hasta que se active la anonimización.
- B. En el orquestador, anonimizando antes del ensamblado del *prompt*, el almacenamiento y el registro, y rechazando los proveedores externos cuando la anonimización está desactivada.
- C. En el *prompt* de sistema: "No repitas nunca datos personales".
- D. En el registro de trazas, borrando las líneas que contengan direcciones de correo.

<details><summary>Respuesta</summary>

**B.** Todas las peticiones pasan por el orquestador, así que una comprobación ahí no se puede
esquivar. Una comprobación en el cliente (A) se la salta cualquier *script* que llame directamente a
la API. Un *prompt* de sistema (C) le pide al modelo que se comporte, pero no impide que los datos se
*envíen*. Limpiar los registros (D) ocurre cuando los datos ya han salido.

</details>

### P2 — Latencia a mano

El perfil `fake-small` tiene 120 ms de sobrecarga, 0,25 ms de prellenado por *token* del *prompt* y
20 ms por cada *token* generado después del primero. Una petición tiene un *prompt* de 120 *tokens* y
una respuesta de 41 *tokens*. Calcula el tiempo hasta el primer *token* (TTFT, *time to first token*),
la latencia total, la velocidad de decodificación y el rendimiento de extremo a extremo.

<details><summary>Respuesta</summary>

- TTFT = 120 + 0,25 × 120 = **150 ms**.
- Total = 150 + 40 × 20 = **950 ms**.
- Velocidad de decodificación = 40 *tokens* ÷ 0,8 s = **50 *tokens*/s**.
- De extremo a extremo = 41 *tokens* ÷ 0,95 s ≈ **43,2 *tokens*/s**.

El número de extremo a extremo es más bajo porque incluye la espera hasta el primer *token*. Por eso
no se debe presentar como "velocidad del modelo".

</details>

### P3 — La factura mensual

La carga de trabajo de la lección 3.1 envía 31.500 peticiones al mes con 180 *tokens* de respuesta
cada una. En `fake-large` (precios de ejemplo: 1,00 € por millón de *tokens* de entrada, 4,00 € por
millón de *tokens* de salida), la factura era de 44,73 € con 700 *tokens* de *prompt* por petición.
Las conversaciones se alargan y el *prompt* medio crece hasta 1.400 *tokens*. ¿Cuál es el nuevo coste
mensual?

<details><summary>Respuesta</summary>

Por petición: (1.400 × 1,00 + 180 × 4,00) ÷ 1.000.000 = 0,00212 €. Al mes: 0,00212 × 31.500 =
**66,78 €**. Duplicar el *prompt* no duplicó la factura, porque la parte de salida (0,00072 € por
petición) no cambió. Aun así, en las conversaciones largas la entrada domina y no deja de crecer.

</details>

### P4 — Encuentra el fallo: un bucle de reintentos

```python
def call_with_retries(provider, messages):
    while True:
        try:
            return provider.complete(messages)
        except Exception:
            time.sleep(1)
```

Nombra al menos tres problemas.

<details><summary>Respuesta</summary>

1. No se rinde nunca: no hay un número máximo de reintentos ni un plazo total.
2. Lo reintenta todo, incluidos los errores que nunca van a salir bien (`bad_request`,
   `context_length`) y los que necesitan otro proveedor (`auth`).
3. Una espera fija de un segundo no tiene espera exponencial ni variación aleatoria (*jitter*), así que
   muchos clientes reintentan en olas sincronizadas.
4. Ignora `Retry-After`, nunca recurre a un respaldo y no registra nada sobre por qué falló la llamada.

Compáralo con `ModelGateway.complete` en el `resilience.py` del lab.

</details>

### P5 — ¿Reintentar o no?

¿Qué fallo **no** debería reintentar la pasarela?

- A. HTTP 429 con `Retry-After: 2`.
- B. HTTP 503.
- C. Un *timeout* después de 10 segundos.
- D. HTTP 400, "maximum context length exceeded".

<details><summary>Respuesta</summary>

**D.** El mismo *prompt* falla igual cada vez; la solución es acortarlo (lección 3.3). A, B y C son
transitorios y merece la pena reintentarlos con espera exponencial; A, después de los dos segundos que
pidió el servidor.

</details>

### P6 — Los reintentos en números

Un proveedor falla en el 20 % de las llamadas, de forma independiente entre ellas. La pasarela permite
hasta 2 reintentos. ¿Qué proporción de peticiones sale bien? ¿Por qué un sistema real podría medir
menos?

<details><summary>Respuesta</summary>

Una petición solo falla si fallan los tres intentos: sale bien el 1 − 0,2³ = **99,2 %**. Los sistemas
reales miden menos cuando los fallos *no* son independientes (una caída hace fallar todos los
intentos, como muestran las filas de caída del lab) y cuando los fallos lentos, como los *timeouts*,
consumen el plazo antes del último reintento.

</details>

### P7 — Presupuesto de entrada

Un modelo tiene una ventana de contexto de 4.096 *tokens*. Pones `max_tokens` a 512, y tu *prompt* de
sistema ocupa 300 *tokens*. ¿Cuántos *tokens* quedan para el historial y el mensaje nuevo?

<details><summary>Respuesta</summary>

Presupuesto de entrada = 4.096 − 512 = 3.584 *tokens*. Menos los 300 del *prompt* de sistema:
**3.284 *tokens*** para el historial y el mensaje nuevo. Olvidarse de restar `max_tokens` es el error
clásico: el *prompt* cabe, y luego la petición falla o la respuesta se corta.

</details>

### P8 — El pedido olvidado

Los chats de soporte largos del *playground* olvidan el número de pedido que dio la persona cliente en
el primer turno. La política `pin_system` está activa. ¿Qué cambias?

- A. Cambiar a una ventana deslizante con los seis últimos mensajes.
- B. Cambiar a la política `summarise` y guardar los números de pedido en un campo estructurado de la sesión.
- C. Subir `max_tokens`.
- D. Nada: pedir al equipo de soporte que repita el número de pedido.

<details><summary>Respuesta</summary>

**B.** En la reproducción del lab, `summarise` todavía conocía `#HW-640183` en el turno 20, a cambio
de 82 *tokens* más que `pin_system`. Una ventana deslizante (A) olvida incluso antes y además descarta
el *prompt* de sistema. Subir `max_tokens` (C) *reduce* el presupuesto de entrada. D traslada el coste
a la clientela.

</details>

### P9 — Encuentra el fallo: el truncado

```python
def build_prompt(system, history, user):
    messages = [system, *history, user]
    return messages[-6:]
```

<details><summary>Respuesta</summary>

El *prompt* de sistema es el mensaje más antiguo, así que es el primero en caerse: a partir del cuarto
turno, el modelo funciona sin sus reglas (17 de 20 turnos en la reproducción del lab). Además, cuenta
mensajes en lugar de *tokens*, así que seis mensajes largos pueden desbordar igualmente la ventana, y
puede partir por la mitad un par `user`/`assistant`. Solución: fija el *prompt* de sistema y llena el
resto con pares completos según el presupuesto de *tokens*.

</details>

### P10 — Leer la tabla del anonimizador

El nivel `regex` obtiene: correos 1224/1224, teléfonos 82/82, nombres 0/82, *recall* 0.941, tickets con
fugas 82. ¿Qué afirmación es correcta?

- A. Una exhaustividad (*recall*) de 0.941 significa que el 94 % de los tickets se pueden enviar sin riesgo.
- B. Todos los tickets con firma siguen filtrando un nombre; la exhaustividad global esconde la omisión completa de un tipo de entidad.
- C. El nivel `regex` tiene una precisión baja.
- D. Los 82 tickets con fugas se deben a las fechas.

<details><summary>Respuesta</summary>

**B.** La exhaustividad cuenta elementos, no tickets, y 1.224 correos fáciles ahogan 82 nombres. La
precisión del nivel `regex` es 1.000 (C es incorrecta); las fechas eran los falsos positivos del nivel
`naive` (D es incorrecta). Informa siempre por tipo de entidad.

</details>

### P11 — "Solo para depurar"

Alguien de tu equipo añade el *prompt* en bruto a cada línea de traza, "temporalmente", para depurar
una respuesta extraña. ¿Qué haces?

<details><summary>Respuesta</summary>

Quitarlo. Los *prompts* en bruto convierten las trazas en una copia no planificada de los datos de la
clientela, sin periodo de retención ni control de acceso, y los campos "temporales" suelen quedarse.
Depura con lo que ya tiene la traza (`prompt_hash`, parámetros, intentos, marcas) y reproduce el caso
con el *prompt* *anonimizado*. Si de verdad hace falta el texto en bruto, acuerda una excepción con
Noor: un almacén aparte, acceso restringido y un periodo de retención corto.

</details>

### P12 🔁 (módulo 02) — Temperatura

En el modo de comparación, `fake-small` dio 1 respuesta distinta en 10 semillas a temperatura 0 y 3 a
temperatura 1,5, y 2 de esas 10 repitieron la política de devoluciones obsoleta de 14 días. Con lo que
aprendiste sobre la decodificación, explica qué hace la temperatura y por qué apareció la respuesta
incorrecta.

<details><summary>Respuesta</summary>

La temperatura divide las puntuaciones del modelo (*logits*) antes del *softmax*. Cerca de 0, la
distribución se concentra en la opción más probable, que es la decodificación voraz (*greedy*). Por
encima de 1 se aplana, así que las opciones poco probables salen elegidas más a menudo. La respuesta
obsoleta siempre estuvo en la distribución con una probabilidad baja; la temperatura 1,5 le dio una
oportunidad real. La temperatura añade variedad, no conocimiento, así que las preguntas factuales se
quedan en 0 y las comprobaciones de salida hacen de red de seguridad.

</details>

### P13 🔁 (módulo 01) — Contar *tokens*

El lab cuenta los *tokens* como palabras más signos de puntuación. ¿Por qué el recuento de un proveedor
para el mismo *prompt* será distinto, y qué recuento debes usar para los presupuestos y la facturación?

<details><summary>Respuesta</summary>

Los modelos reales usan tokenizadores de subpalabras (como la codificación por pares de bytes, *byte-pair encoding*)
aprendidos de los datos de entrenamiento: las palabras raras, los números, el texto en español y el
código suelen dividirse en varios *tokens*, y las plantillas de chat añaden *tokens* especiales para
los roles. Usa el tokenizador del propio modelo, o el uso que devuelve el proveedor con cada respuesta,
para los presupuestos y el coste. La aproximación del lab solo sirve para enseñar la forma del
crecimiento.

</details>

## Guía de puntuación

| Respuestas correctas | Qué hacer a continuación |
|---|---|
| 12–13 | Tienes todo a punto para el módulo 04. |
| 9–11 | Vuelve a leer las secciones de las preguntas que fallaste y repite el paso del lab correspondiente. |
| 0–8 | Repite los pasos 1 a 5 del lab y vuelve a hacer el quiz dentro de uno o dos días. |

## ⏭️ Siguiente

[Módulo 04 — Adaptar LLM: *prompting*, contexto y *fine-tuning*](../04-prompting-context-and-fine-tuning/README.es.md)
