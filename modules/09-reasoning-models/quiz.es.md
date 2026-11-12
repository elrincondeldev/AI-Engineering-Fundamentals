🌐 [English](quiz.md) · Español

# Módulo 09 — Quiz

> 13 preguntas, unos 25 minutos. Responde primero en papel y después abre las respuestas. Las
> preguntas marcadas con 🔁 repasan módulos anteriores.

### P1 — Tokens de pensamiento

Una API de razonamiento devuelve una respuesta corta y oculta su razonamiento. ¿Qué afirmación es
verdadera?

- A. Los *tokens* de razonamiento ocultos son gratis; solo se facturan los *tokens* visibles.
- B. Los *tokens* de razonamiento ocultos son *tokens* de salida generados: cuestan dinero y tiempo, y los datos de uso informan de ellos.
- C. Los *tokens* de razonamiento ocultos se cuentan como *tokens* del *prompt*.
- D. Los *tokens* de razonamiento solo existen cuando fijas un esfuerzo de razonamiento alto.

<details><summary>Respuesta</summary>

**B.** Los *tokens* de pensamiento se generan como cualquier salida. Los pagas y los esperas aunque
nunca los veas, así que presupuesta a partir de los datos de uso, no de la longitud de la respuesta
visible.

</details>

### P2 — El precio de un punto

En `uv run lab solve`, la cadena de pensamiento obtiene 49.4% con 72.7 *tokens* por respuesta, y el
pensamiento con esfuerzo alto obtiene 71.9% con 194.1 *tokens* por respuesta. ¿Cuántos *tokens* extra
por respuesta cuesta cada punto porcentual extra?

<details><summary>Respuesta</summary>

(194.1 − 72.7) / (71.9 − 49.4) = 121.4 / 22.5 ≈ **5.4 *tokens* por punto**. Las medias ocultan el
reparto: en los problemas del hub de cinco pasos, el mismo cambio gana 40 puntos; en los de dos pasos,
13,2.

</details>

### P3 — El estancamiento

La votación por mayoría sobre los problemas numéricos llega a 89.1% con N = 15 y sigue en 89.1% con
N = 45, mientras que pass@15 ya es 100%. ¿Cuál es la mejor explicación?

- A. 45 muestras son pocas; con N = 200 la votación llegaría al 100 %.
- B. La votación falla en los problemas en los que la misma respuesta errónea (una trampa) es la más frecuente, y más muestras hacen que esa respuesta gane de forma más fiable.
- C. La regla de desempate está mal.
- D. pass@N se calcula de forma incorrecta.

<details><summary>Respuesta</summary>

**B.** Los 12 problemas que falla la votación con N = 45 tienen como respuesta más votada una respuesta
de trampa. La votación elimina los errores independientes y amplifica los correlacionados. A es la
respuesta errónea tentadora: más muestras solo afinan la estimación de la respuesta más frecuente, que
en esos problemas es errónea.

</details>

### P4 — Encuentra el fallo

```python
def vote(answers: list[str]) -> str:
    return Counter(answers).most_common(1)[0][0]

vote(["175.1", "175.10", "90"])
```

¿Por qué esta votación puede devolver la respuesta errónea para un pedido con un total de 175,10 €?

<details><summary>Respuesta</summary>

Cuenta las cadenas en bruto, así que `175.1` y `175.10` son dos respuestas distintas con un voto cada
una, empatadas con `90`; gana la que aparezca primero. Normaliza las respuestas antes de contar (el
`answer_key` del lab convierte las dos en la misma fracción exacta) y haz explícita la regla de
desempate.

</details>

### P5 — pass@k a mano

Sacaste n = 5 muestras para un problema y c = 1 es correcta. Usando el estimador insesgado
pass@k = 1 − C(n − c, k) / C(n, k), ¿cuánto vale pass@2?

<details><summary>Respuesta</summary>

C(4, 2) = 6 y C(5, 2) = 10, así que pass@2 = 1 − 6/10 = **0.4**. Intuición: de las 10 formas de elegir
2 de las 5 muestras, 4 incluyen la correcta.

</details>

### P6 — ¿ORM o PRM?

En las 4.950 muestras numéricas, el ORM simulado deja pasar 571 respuestas erróneas y rechaza 518
correctas; el PRM simulado deja pasar 64 y rechaza 906. Quieres ordenar 9 muestras de la misma
pregunta y quedarte con una. ¿Cuál prefieres, y por qué podrías preferir el otro como filtro sobre un
flujo de respuestas sueltas?

<details><summary>Respuesta</summary>

Para ordenar, el PRM: puede ver las lecturas erróneas, y *best-of-9* con el PRM llegó a 95.5% frente a
86.4% con el ORM. Ordenar solo necesita que la respuesta correcta puntúe por encima de las erróneas
*para la misma pregunta*. Como filtro de aprobado o suspenso sobre respuestas sueltas, los 906 falsos
rechazos del PRM tirarían mucho trabajo bueno; tendrías que ajustar su umbral o combinarlo con una
comprobación exacta.

</details>

### P7 — Ventajas relativas al grupo

Un grupo de GRPO de cuatro respuestas recibe las recompensas 1, 0, 0, 0. Calcula las ventajas
(desviación típica poblacional).

<details><summary>Respuesta</summary>

Media = 0.25. Desviación típica = √((0.75² + 3 × 0.25²) / 4) = √0.1875 ≈ 0.433. Ventajas:
0.75 / 0.433 ≈ **1.73** para la respuesta correcta y −0.25 / 0.433 ≈ **−0.58** para cada una de las
erróneas. Suman cero. Con las recompensas 1, 1, 1, 1, todas las ventajas serían 0 y el grupo no
enseñaría nada.

</details>

### P8 — ¿Qué harías? Acertijos con presupuesto

Jordan necesita resolver los acertijos numéricos, y el límite de Sam es de unos 500 *tokens* de
*prompt* y 7 llamadas por acertijo. Según el lab: muestrear hasta verificar con hasta 15 intentos
resuelve 41/50 (485.7 *tokens* de *prompt*, 6.2 llamadas); ToT BFS con anchura 5 resuelve 36/50
(2,381.9 *tokens* de *prompt*, 28.6 llamadas). ¿Cuál lanzas?

<details><summary>Respuesta</summary>

Muestrear hasta verificar. Cumple los dos límites y resuelve más acertijos, porque el verificador
exacto hace que cada muestra extra sea segura. El ToT gana en *tokens* generados, pero cada llamada de
propuesta y de evaluación reenvía el acertijo, así que se sale de los presupuestos de *tokens* de
*prompt* y de llamadas.

</details>

### P9 — Encuentra el fallo: un resultado de STaR sospechoso

El bucle STaR de alguien del equipo muestra, tras dos rondas, un salto mucho mayor que cualquiera de la
tabla del lab. Su evaluación llama a `evaluate(config, train)`. ¿Qué está mal?

<details><summary>Respuesta</summary>

Evalúa con los problemas de entrenamiento, con cuyas justificaciones se acaba de ajustar el modelo:
fuga de datos (*data leakage*). Evalúa con problemas reservados que nunca entren en el bucle, como hace
el lab con los ids pares. El número honesto sobre el conjunto reservado en el lab es 72.5% tras cuatro
rondas.

</details>

### P10 — ¿Qué harías? Señales de explotación de la recompensa

Tras cambiar *best-of-N* a un nuevo verificador aprendido, la puntuación media que da el verificador a
las respuestas elegidas sube cada semana, las respuestas elegidas son cada vez más largas y la
exactitud que miden personas en revisiones puntuales no se mueve. ¿Qué sospechas y qué haces?

<details><summary>Respuesta</summary>

Explotación de la recompensa (*reward hacking*): la selección premia algo que le gusta al verificador
(aquí, la longitud) en lugar de la corrección, como hizo el ORM con sesgo de longitud de la lección
9.3. Mide contra la verdad de referencia en un conjunto reservado, busca el sesgo (puntuación frente a
longitud), añade comprobaciones exactas donde sea posible y corrige o sustituye el verificador antes
de fiarte de sus puntuaciones.

</details>

### P11 🔁 (módulo 02)

Ejecutas la autoconsistencia con 9 muestras por pregunta, pero todas las muestras son idénticas y la
votación nunca ayuda. ¿Qué ajuste de decodificación del módulo 02 es la causa probable, y por qué?

<details><summary>Respuesta</summary>

La decodificación voraz (*greedy*, temperatura 0): el modelo elige siempre el *token* más probable, así
que todas las muestras son la misma cadena y la votación no tiene entre qué elegir. La autoconsistencia
necesita muestras diversas: usa una temperatura mayor que cero (a menudo con *top-p*) para que
aparezcan cadenas distintas, y errores distintos.

</details>

### P12 🔁 (módulo 04)

Jordan aprueba destilar trazas de razonamiento verificadas en un modelo abierto pequeño, con una sola
GPU modesta. ¿Qué enfoque de ajuste fino del módulo 04 encaja, y qué cambia en el modelo?

<details><summary>Respuesta</summary>

Un método eficiente en parámetros como LoRA (o QLoRA, que además cuantiza el modelo base congelado).
Congela los pesos originales y entrena pequeñas matrices de bajo rango añadidas a algunas capas, así
que las necesidades de memoria y de cómputo bajan mucho y el adaptador es lo bastante pequeño para
guardarlo e intercambiarlo. Los datos de entrenamiento siguen siendo datos de SFT normales: pregunta,
traza de razonamiento, respuesta.

</details>

### P13 🔁 (módulo 08)

En el módulo 08 usaste la búsqueda en árbol para un agente web. Nombra una cosa que haga la búsqueda en
árbol más difícil para un agente que actúa en la web que el árbol de pensamientos en los acertijos de
número objetivo.

<details><summary>Respuesta</summary>

Cualquiera de estas: las acciones pueden tener efectos secundarios y no siempre se pueden deshacer, así
que retroceder no sale gratis; el estado es una página web o una conversación entera, no una lista
corta de números, así que es caro de copiar y de evaluar; y rara vez hay un verificador exacto para la
respuesta final, así que la búsqueda solo se guía por juicios aprendidos o heurísticos.

</details>

## 📊 Guía de puntuación

| Puntuación | Qué significa |
|---|---|
| 11–13 | Tienes todo a punto para el módulo 10. |
| 8–10 | Vuelve a leer las autoevaluaciones de las lecciones en las que fallaste preguntas y sigue adelante. |
| 0–7 | Vuelve a hacer el lab, sobre todo los pasos 3 a 7, antes de seguir. |

## ⏭️ Siguiente

[Módulo 10 — *Deep research* y despliegue local](../10-deep-research-and-local-deployment/README.es.md)
