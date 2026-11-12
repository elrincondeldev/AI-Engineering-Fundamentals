🌐 [English](quiz.md) · Español

# Quiz del módulo 05 — RAG I: recuperación

Trece preguntas. Responde cada una en papel antes de abrir la respuesta. Las preguntas marcadas con 🔁
repasan módulos anteriores.

### P1

¿Cuál de estos problemas **no** resuelve por sí sola la generación aumentada por recuperación (RAG)?

- A. El modelo nunca ha oído hablar de un producto lanzado después de que se recogieran sus datos de
  entrenamiento.
- B. La política de devoluciones cambió el mes pasado.
- C. Las respuestas del modelo son demasiado largas y demasiado formales.
- D. La clientela quiere ver de qué artículo sale una respuesta.

<details><summary>Respuesta</summary>

**C.** El tono y la longitud son comportamiento; se moldean con *prompting* o *fine-tuning*. RAG
resuelve la fecha de corte del conocimiento (A), los hechos que cambian (B) y las citas (D) poniendo en
el *prompt* pasajes elegidos y actualizados.

</details>

### P2

Sin filtros, BM25 sitúa primero la política de devoluciones sustituida de 2023 (`kb-010`, "14 días")
para "What is the return window for Hearthwise products?". ¿Cuál es la mejor solución?

- A. Añadir "prefiere siempre la política más reciente" al *prompt* de sistema.
- B. Descartar antes de indexar los documentos cuyo `status` no sea `current`.
- C. Recuperar 20 fragmentos en lugar de 5 para que también se incluya la política vigente.
- D. Cambiar BM25 por *embeddings*.

<details><summary>Respuesta</summary>

**B.** El documento obsoleto no debería llegar nunca al índice. Una regla en el *prompt* (A) pide al
modelo que resuelva un conflicto que podrías haber eliminado; más fragmentos (C) mantienen el
equivocado en el contexto; los *embeddings* (D) también encontrarían la política antigua, porque trata
del mismo tema.

</details>

### P3

La palabra "watcher" aparece en 4 de los 111 fragmentos por encabezado. Calcula su frecuencia inversa
de documento en BM25, `ln(1 + (N - df + 0.5) / (df + 0.5))`, con dos decimales.

<details><summary>Respuesta</summary>

`ln(1 + (111 - 4 + 0.5) / (4 + 0.5)) = ln(1 + 107.5 / 4.5) = ln(24.89) = 3.21`. Como comparación, "the"
(en 75 fragmentos) tiene una IDF de 0.39: los términos raros son los que aportan la puntuación.

</details>

### P4

Encuentra el fallo. El *recall@5* bajó de 0.944 a 0.861 después de este cambio en `BM25.scores`:

```python
def scores(self, query: str) -> list[float]:
    out = [0.0] * self.index.n_docs
    for term in query.split():
        ...
```

<details><summary>Respuesta</summary>

El índice se construyó con `tokenize` (minúsculas, sin puntuación), pero la consulta se divide por
espacios. "Watcher" ya no coincide con "watcher" y "weigh?" no coincide con nada. Los documentos y las
consultas tienen que usar el mismo tokenizador: `for term in tokenize(query, fold=self.fold):`.

</details>

### P5

Fusión por rango recíproco con `k = 60`. BM25 sitúa el fragmento X en segundo lugar y no aparece en la
lista vectorial. El fragmento Y está en el décimo puesto en las dos listas. ¿Cuál queda más arriba
después de la fusión?

<details><summary>Respuesta</summary>

**Y.** X puntúa 1 / 62 = 0.0161. Y puntúa 1 / 70 + 1 / 70 = 0.0286. La RRF premia el acuerdo, y por eso
el fragmento de envíos en español cayó del puesto 2 (BM25) al 31 en la búsqueda híbrida del lab.

</details>

### P6

¿Qué fragmentador escribe cada valor de una tabla junto al nombre de su columna?

- A. De tamaño fijo con solapamiento
- B. Recursivo
- C. Semántico
- D. Por filas de tabla

<details><summary>Respuesta</summary>

**D.** Cada fila de una tabla ancha se convierte en un fragmento como `Compare Hearthwise devices | Watcher S1 | ...;
Weight (g): 34; ...`. Los fragmentos de tamaño fijo y los recursivos también encontraron el documento
de la tabla en el lab, pero contienen "... yes 34 -10 to 45 ..." con la fila de cabecera en otro
fragmento.

</details>

### P7

Alguien del equipo ve `bm25/fixed` con un *recall@5* de 0.972 y 1.00 en la categoría `table`, por
encima de `bm25/table` (0.944), y quiere cambiar. ¿Qué compruebas antes de darle la razón?

<details><summary>Respuesta</summary>

Lee los fragmentos que recibiría el generador. El fragmento de tamaño fijo con la fila del Watcher S1 no
tiene nombres de columna, así que un modelo tiene que adivinar qué significa "34". Comprueba también el
tamaño de la diferencia: con 36 preguntas, una pregunta vale unos 0.028, así que 0.972 frente a 0.944 es
una sola pregunta. Una decisión justa sopesa el *recall* frente a la legibilidad de los fragmentos e,
idealmente, ejecuta en los dos la evaluación de respuestas del módulo 06.

</details>

### P8

En el *benchmark* del lab, el índice HNSW ocupa 26.41 MB, mientras que los vectores en bruto ocupan
12.80 MB. ¿Por qué?

- A. HNSW guarda cada vector dos veces.
- B. HNSW guarda un grafo: hasta 2 × M enlaces a vecinos por vector en su capa inferior, más las capas
  superiores.
- C. FAISS rellena los vectores de HNSW hasta 128 dimensiones.
- D. HNSW guarda los centroides de k-means de cada capa.

<details><summary>Respuesta</summary>

**B.** Con M = 32, cada vector guarda hasta 64 enlaces de 4 bytes en la capa inferior, más o menos lo
mismo que el propio vector de 64 dimensiones. Ese es el precio del *recall* alto de HNSW con poco
trabajo por consulta.

</details>

### P9

IVF con 256 listas da un *recall@10* de 0.741 con nprobe = 1, de 0.917 con nprobe = 8 y de 0.977 con
nprobe = 32. El objetivo de Sam es un *recall@10* de al menos 0.97 con el menor trabajo posible por
consulta. ¿Qué configuración eliges y cuánto cuesta en comparación con nprobe = 8?

<details><summary>Respuesta</summary>

nprobe = 32: el único ajuste de IVF por encima de 0.97. Calcula 6.504 similitudes por consulta frente a
1.814 con nprobe = 8, unas 3.6 veces el trabajo, aunque sigue muy por debajo de las 50.000 de la
búsqueda exacta. HNSW con efSearch = 128 (0.981 con 3.055 similitudes) es la alternativa si el doble de
memoria es aceptable.

</details>

### P10

El grafo de conocimiento responde "What is the price of the Lookout C2 in Mexico?" con €89. El centro
de ayuda nunca indica un precio en México. ¿Qué harías?

<details><summary>Respuesta</summary>

Envía una pregunta al grafo solo cuando cada entidad y restricción que contiene encaje en el esquema.
"In Mexico" no encaja, así que recurre a la recuperación de texto y deja que el generador diga que no lo
sabe (q037 no tiene respuesta a propósito). No devuelvas nunca un hecho del grafo que responde a una
pregunta más sencilla que la que se hizo.

</details>

### P11 🔁 (módulo 02)

Después de muchos experimentos con las 36 preguntas de evaluación, una configuración alcanza un
*recall@5* de 1.000. En el lenguaje de la evaluación de modelos, ¿qué salió mal y cuál es la solución?

<details><summary>Respuesta</summary>

El conjunto de evaluación se convirtió en un conjunto de ajuste: las decisiones se tomaron después de
mirar sus fallos, así que la puntuación es optimista, el mismo problema que un *benchmark* que se coló
en el entrenamiento. La solución es un conjunto nuevo y reservado (preguntas nuevas, idealmente de
tickets reales) que nadie haya mirado mientras construía el flujo.

</details>

### P12 🔁 (módulo 03)

En el *playground* de LLM, las conversaciones largas acaban superando la ventana de contexto del
modelo. Nombra dos formas de gestionarlo y explica por qué los pasajes recuperados hacen el problema
más urgente en un chatbot RAG.

<details><summary>Respuesta</summary>

Descartar o truncar los turnos más antiguos (una ventana deslizante), o resumir los turnos anteriores en
una memoria corta. En un chatbot RAG, los pasajes recuperados, las reglas del sistema y la conversación
comparten la misma ventana, así que cada fragmento que recuperas deja menos sitio para el historial.
Recuperar menos fragmentos y mejores (reordenación, fragmentos por filas de tabla que llevan su propio
contexto) también es una decisión sobre el presupuesto de contexto.

</details>

### P13 🔁 (módulo 04)

¿Qué afirmación sobre el *fine-tuning* con LoRA es verdadera?

- A. Actualiza todos los pesos del modelo, pero con una tasa de aprendizaje más pequeña.
- B. Congela el modelo y entrena pequeñas matrices de bajo rango que se suman a algunas matrices de
  pesos.
- C. No necesita datos de entrenamiento, solo un *prompt* de sistema.
- D. Es la mejor forma de mantener actualizado lo que sabe un modelo sobre una política de devoluciones
  que cambia.

<details><summary>Respuesta</summary>

**B.** Consulta el [artículo de LoRA](https://arxiv.org/abs/2106.09685). D es la trampa: el
*fine-tuning* moldea el comportamiento, y un hecho aprendido en los pesos se queda ahí, obsoleto, hasta
que vuelves a entrenar. Los hechos que cambian van en un índice de recuperación.

</details>

## 📊 Guía de puntuación

| Puntuación | Qué significa | Qué hacer |
|---|---|---|
| 11–13 | Sabes construir y defender un flujo de recuperación | Pasa al módulo 06 |
| 8–10 | Las ideas están ahí; algunos detalles flojean | Relee las lecciones de las preguntas que fallaste y repite el paso del lab correspondiente |
| 0–7 | El flujo todavía no está claro | Repite las lecciones 5.2 y 5.3 con el lab abierto y vuelve a hacer el quiz |

## ⏭️ Siguiente

[Módulo 06 — RAG II: el chatbot de soporte](../06-rag-generation-and-evaluation/README.es.md): convierte
el recuperador en un chatbot fundamentado que cita, se niega cuando debe y resiste las instrucciones
inyectadas.
