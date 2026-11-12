🌐 [English](quiz.md) · Español

# Quiz del módulo 02 — Del modelo base al asistente

Responde a cada pregunta antes de abrir la solución. Apunta tus respuestas; explicar el *porqué* cuenta
tanto como la letra o el número. Las preguntas marcadas con 🔁 repasan el módulo 01.

### P1

Bajas la temperatura de muestreo de 1,0 a 0,5. ¿Qué le pasa a la distribución del siguiente *token*?

- A. Todos los *tokens* pasan a ser igual de probables
- B. Los *tokens* más probables se llevan más probabilidad y la cola, menos
- C. Se añade ruido aleatorio a los *logits*
- D. No cambia nada hasta que la temperatura llega a 0

<details><summary>Respuesta</summary>

**B.** La temperatura divide los *logits* antes de *softmax*; dividir por 0,5 duplica las distancias
entre puntuaciones. En el lab, P("device") tras "to reset the" pasa de 0,612 con T = 1,0 a 0,996 con
T = 0,5. La temperatura no añade ruido (C), y una temperatura alta aplana la distribución pero nunca la
vuelve exactamente uniforme (A).

</details>

### P2

Las probabilidades del siguiente *token*, ordenadas, son 0,5, 0,3, 0,15 y 0,05. ¿Cuántos *tokens*
conserva *top-p* con p = 0,8? ¿Y con p = 0,81?

<details><summary>Respuesta</summary>

**2 y 3.** Las sumas acumuladas son 0,5, 0,8, 0,95 y 1,0. *Top-p* conserva el prefijo más corto cuya suma
alcanza p: 0,8 se alcanza tras dos *tokens*; 0,81 necesita el tercero. El test del lab
`test_top_p_keeps_smallest_set_reaching_p` comprueba exactamente estos casos.

</details>

### P3

¿Para qué tarea es la búsqueda en haz la opción más sensata?

- A. Un chatbot amable que responde preguntas abiertas
- B. Una lluvia de ideas de nombres de producto
- C. Traducir del inglés al español una etiqueta de producto corta
- D. Escribir correos de marketing variados

<details><summary>Respuesta</summary>

**C.** La búsqueda en haz busca la única salida más probable, lo que encaja con tareas con una respuesta
correcta. Con *prompts* abiertos produce texto corto y genérico: en el lab, "to reset the" se convierte
en "device keeps working during the holidays." y el *benchmark* muestra solo 6 salidas distintas de 30.

</details>

### P4

En producción, el 30 % de las respuestas del asistente terminan con `finish_reason = "length"`. ¿Qué
haces primero?

<details><summary>Respuesta</summary>

Lee una muestra de esas respuestas. La mayoría están cortadas a mitad de frase, y quien las recibe las ve
como respuestas rotas. Después busca la causa: las respuestas se alargaron tras un cambio de *prompt*, el
modelo dejó de emitir su *token* de fin de turno, un bucle o un límite de *tokens* simplemente demasiado
bajo. Arregla la causa (secuencias de parada, *prompt*, penalización por repetición, modelo) y mantén la
proporción de finales por `length` en un panel. Bajar aún más el máximo de *tokens* lo empeoraría.

</details>

### P5 — encuentra el fallo

```python
def loss_mask(tokens):
    mask, in_assistant = [], False
    for tok in tokens:
        if tok in ("<|system|>", "<|user|>", "<|assistant|>"):
            in_assistant = tok == "<|assistant|>"
            mask.append(0)
        elif tok == "<|end|>":
            mask.append(0)
            in_assistant = False
        else:
            mask.append(1)
    return mask
```

<details><summary>Respuesta</summary>

Hay dos fallos. La rama `else` añade 1 para cada *token* normal, así que también se entrenan el *prompt*
de sistema y el mensaje del usuario; debería añadir `1 if in_assistant else 0`. Y el marcador de fin de
un turno del asistente recibe 0, así que el modelo nunca aprende a parar; debería recibir
`1 if in_assistant else 0`.

</details>

### P6

El limpiador de SFT señala 11 filas como contaminadas con el conjunto de evaluación, pero las etiquetas
solo marcan 5. ¿Qué explica las otras 6?

- A. El umbral de n-gramas es demasiado bajo, así que son falsas alarmas
- B. Copian palabra por palabra la respuesta de referencia de la pregunta de evaluación q027, una fuga real que las etiquetas pasaron por alto
- C. Son filas en español que la comprobación de idioma etiquetó mal
- D. Son duplicados exactos de las 5 filas etiquetadas

<details><summary>Respuesta</summary>

**B.** Las 6 contienen "In the app under Account > Orders: tap the order and choose Download invoice.",
la respuesta de referencia de q027. Una precisión de 0,45 frente a las etiquetas esconde una detección
correcta. Lee los desacuerdos antes de fiarte de la comprobación o de las etiquetas.

</details>

### P7

Un modelo de recompensa de Bradley–Terry puntúa la respuesta A con r = 1,2 y la respuesta B con
r = 0,2. ¿Qué probabilidad da a que las personas prefieran A?

<details><summary>Respuesta</summary>

**Aproximadamente 0,731.** P(A frente a B) = σ(1,2 − 0,2) = σ(1,0) = 1 / (1 + e^(−1)) ≈ 0,731. Solo
importa la diferencia: 5,2 frente a 4,2 daría la misma probabilidad.

</details>

### P8

Un modelo de recompensa de palabras más longitud obtiene 1,000 en los pares de entrenamiento y 1,000 en
la partición de prueba oficial. ¿Cuál es la explicación más probable?

- A. El modelo ha aprendido a la perfección las preferencias de los agentes
- B. Las mismas preguntas aparecen en ambas particiones, así que el modelo memorizó qué respuesta va con qué pregunta
- C. La partición de prueba es demasiado fácil porque tiene menos pares
- D. El sesgo de longitud se anula en la partición de prueba

<details><summary>Respuesta</summary>

**B.** Las 23 preguntas se repiten entre las particiones. Con una partición disjunta por *prompts*, el
mismo tipo de modelo cae a 0,714, y entre sus pesos más grandes está el rasgo de longitud. Las
puntuaciones perfectas con datos reservados son motivo para buscar fugas.

</details>

### P9

En un RLHF basado en PPO, el modelo de recompensa da r = 1,5, la log-probabilidad de la respuesta según
la política es −10,0, la del modelo de referencia es −14,0, y β = 0,2. ¿Qué recompensa optimiza PPO para
esta muestra?

<details><summary>Respuesta</summary>

**0,7.** r − β · (log π − log π_ref) = 1,5 − 0,2 × (−10,0 − (−14,0)) = 1,5 − 0,8 = 0,7. La política ha
derivado 4 nats hacia esta respuesta, y la penalización KL cobra por esa deriva.

</details>

### P10

Al principio del entrenamiento con DPO, la política es igual al modelo de referencia. ¿Cuál es la pérdida
de DPO de cada par y por qué?

<details><summary>Respuesta</summary>

**ln 2 ≈ 0,693.** El margen (el log-cociente de la política para la elegida menos el de la rechazada) es
0, y −log σ(0) = −log 0,5 = ln 2. Después, el entrenamiento baja la pérdida ensanchando el margen, lo que
puede ocurrir incluso mientras baja la probabilidad de la respuesta elegida, siempre que la rechazada
baje más.

</details>

### P11

La respuesta de referencia es "30 days from delivery." ¿Qué candidata obtiene el ROUGE-L más alto?

- A. "You have 30 days after delivery to return it."
- B. "Thirty days, counted from the day it arrives."
- C. "14 days from delivery."
- D. "Returns are accepted for 30 days from delivery, opened or not."

<details><summary>Respuesta</summary>

**C**, la única respuesta incorrecta, con 0,80 en la tabla de métricas del lab, porque comparte tres de
sus cuatro palabras con la referencia. Las métricas de solapamiento miden la redacción, no la corrección;
por eso rag_eval también incluye cadenas `must_not_contain` como "14 days".

</details>

### P12 — ¿qué harías?

Jordan quiere usar un juez LLM para comparar cada semana los modelos de dos proveedores, preguntándole
una vez por par "Which answer is better, A or B?" con el proveedor X siempre como A. ¿Qué cambias?

<details><summary>Respuesta</summary>

Aleatoriza o intercambia el orden y pregunta dos veces por par, contando solo los veredictos coherentes,
porque los jueces tienen sesgo de posición (el juez del lab coincide con las personas 1,000 de las veces
con la mejor respuesta primero y 0,667 con ella en segundo lugar). Sigue con qué frecuencia gana la
respuesta más larga (sesgo de verbosidad), usa un juez de una familia distinta a la de ambos proveedores
(autopreferencia), calíbralo con pares de Hearthwise etiquetados por personas e informa de intervalos, no
solo de tasas de victoria.

</details>

### P13 🔁 (módulo 01)

Dos páginas de studyhub.example están en un inglés limpio y bien escrito, sin *spam* y sin datos
personales. ¿Por qué debería descartarlas igualmente un *pipeline* de limpieza para preentrenamiento?

<details><summary>Respuesta</summary>

Copian material de evaluación: 6 preguntas de rag_eval con sus respuestas y 6 problemas de reasoning con
sus soluciones. Entrenar con ellas permitiría a un modelo recordar las respuestas e inflar sus
puntuaciones (contaminación). Su etiqueta `keep` en `datasets/web/labels.jsonl` es falsa por ese motivo,
y el detector de contaminación de la lección 2.4 encuentra las dos.

</details>

### P14 🔁 (módulo 01)

En el rastreo web, los pares de casi duplicados etiquetados tienen una similitud de Jaccard por 5-gramas
de palabras de al menos 0,969, y dos páginas conservadas distintas nunca superan 0,396. ¿Qué umbral
elegirías para eliminar casi duplicados en este rastreo, y qué riesgo tiene elegir 0,3?

<details><summary>Respuesta</summary>

Cualquier valor holgadamente entre los dos, por ejemplo 0,8, los separa en este rastreo. Con 0,3
eliminarías también páginas distintas que solo comparten expresiones comunes (dos páginas conservadas
llegan a 0,396) y perderías datos buenos. El mismo razonamiento de precisión y *recall* se aplica al
filtro de duplicados de SFT de la lección 2.2.

</details>

### P15 🔁 (módulo 01)

El modelo X informa de una perplejidad de 8 en tus correos de soporte, y el modelo Y, de 12. ¿Puedes
concluir que X predice mejor los correos?

<details><summary>Respuesta</summary>

No, a menos que usen el mismo tokenizador. La perplejidad es por *token*, y un tokenizador que divide el
texto en más trozos y más pequeños hace más fácil cada predicción. Compara con la misma tokenización, o
normaliza por carácter o por byte. En el lab, las páginas en español parecen solo moderadamente
sorprendentes (20,30) porque el 80 % de sus palabras se colapsan en `<unk>`: el vocabulario da forma al
número.

</details>

## Guía de puntuación

| Respuestas correctas | Qué significa |
|---|---|
| 13–15 | Puedes pasar al módulo 03. |
| 10–12 | Bien. Relee la autoevaluación de las lecciones de las preguntas que fallaste. |
| 7–9 | Repite los pasos del lab relacionados y los puntos de "Rómpelo" antes de seguir. |
| 0–6 | Repasa las lecciones en orden; las salidas del lab son el mejor punto de apoyo. |

## ⏭️ Siguiente

[Módulo 03 — Construye un playground de LLM →](../03-llm-playground/README.es.md)

---

[← Anterior: Lab](lab/README.es.md) · [🏠 Módulo 02](README.es.md) · [Siguiente: Chuleta →](cheatsheet.es.md)
