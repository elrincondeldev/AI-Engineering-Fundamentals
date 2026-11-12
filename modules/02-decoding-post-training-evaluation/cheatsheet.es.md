🌐 [English](cheatsheet.md) · Español

# Chuleta del módulo 02 — Del modelo base al asistente

## El camino

```text
modelo base → SFT (formato, tono, parada) → optimización de preferencias (RLHF con PPO, o DPO) → asistente
en cada paso: la decodificación convierte logits en texto · la evaluación decide si funcionó
```

## Decodificación

| Ajuste | Regla | Úsalo para |
|---|---|---|
| Temperatura T | softmax(logits / T); T < 1 afila, T > 1 aplana | Creatividad frente a estabilidad |
| Voraz (*greedy*) | argmax en cada paso; determinista | Extracción, clasificación, JSON |
| Búsqueda en haz (k haces) | Conserva las k mejores secuencias parciales por suma de log-probabilidades | Tareas con una respuesta correcta (traducción); sosa con texto abierto |
| *Top-k* | Conserva los k *tokens* más probables | Corte simple de la cola; no se adapta |
| *Top-p* (núcleo) | Conserva el conjunto más pequeño con probabilidad acumulada ≥ p | Corte adaptativo de la cola |
| Penalización por repetición | *Token* ya visto: *logit* positivo / penalización, *logit* negativo × penalización | Romper bucles |
| Frecuencia / presencia | Resta α × recuento, o β una vez presente | Control de bucles al estilo API |
| Secuencias de parada | Termina cuando el texto contiene la cadena (que no se incluye) | Finales limpios |
| Máximo de *tokens* | Techo rígido; `finish_reason = "length"` | Protección de coste y latencia, no un arreglo para la divagación |
| Semilla | La misma semilla y los mismos ajustes dan la misma muestra (en la medida de lo posible en producción) | Reproducibilidad |

Orden de los procesadores en el lab: penalizaciones → *top-k* → *top-p* → *softmax* a T → extracción.

Por qué los modelos se enrollan: sin fin de turno aprendido (arreglo: SFT), sin secuencia de parada, máximo de *tokens* alcanzado, bucles de repetición.

## Ajuste fino supervisado

- Plantilla de chat: `<|role|>\ncontent<|end|>\n` por turno; *prompt* de generación = `<|assistant|>\n` abierto.
- Máscara de pérdida: 1 en los *tokens* del asistente **y** en su `<|end|>`; 0 en sistema, usuario y marcadores de rol.
- El SFT enseña formato, tono, hábitos y cuándo parar; es una mala forma de enseñar datos que cambian.
- Auditoría de datos: vacías, baja calidad, idioma incorrecto, PII, contaminación de la evaluación, datos desactualizados, duplicados.
- Precisión = señaladas correctas / señaladas; *recall* = señaladas correctas / filas etiquetadas. Lee los desacuerdos.

## Optimización de preferencias

| Idea | Fórmula |
|---|---|
| Bradley–Terry | P(A frente a B) = σ(r(A) − r(B)); pérdida = −log σ(r(elegida) − r(rechazada)) |
| Recompensa al estilo PPO | r(x, y) − β · (log π(y dado x) − log π_ref(y dado x)) |
| Recorte de PPO | min(ρ · A, clip(ρ, 1 − ε, 1 + ε) · A), con ρ = π_nueva / π_antigua |
| Pérdida de DPO | −log σ(β · [(log π(c) − log π_ref(c)) − (log π(r) − log π_ref(r))]); vale ln 2 al principio |
| Recompensa implícita (DPO) | β · (log π(y) − log π_ref(y)) |

- Recompensas verificables: un programa comprueba la respuesta (matemáticas, tests, esquemas). Busca resquicios en el verificador.
- Evalúa los modelos de recompensa con ***prompts* nuevos** y frente a "gana la respuesta más larga".
- Explotación de la recompensa: la política se aprovecha de la recompensa (relleno, adulación). Vigila la longitud, la KL y las muestras.

## Evaluación

| Método | Mide | Trampa principal |
|---|---|---|
| EM / F1 por *tokens* | Coincidencia de la cadena normalizada o solapamiento de palabras | Las paráfrasis correctas puntúan bajo |
| BLEU / ROUGE-L | Precisión de n-gramas / subsecuencia común más larga | Una respuesta incorrecta con las palabras adecuadas puntúa alto |
| Perplejidad | exp(log-verosimilitud negativa media) sobre texto reservado | Previsible ≠ bueno; depende del tokenizador |
| *Benchmarks* (MMLU, GSM8K, HumanEval) | Tareas fijas con puntuación automática | Contaminación; puede no corresponder a tu tarea |
| Votos humanos por pares | Qué respuesta prefieren las personas | Necesita muchos votos; informa de intervalos |
| Elo | Actualizaciones de puntuación en línea (K × (resultado − esperado)) | Depende del orden de los votos |
| Puntuaciones Bradley–Terry | Una fuerza por modelo a partir de todos los votos | Necesita suficientes votos por par |
| LLM como juez | Veredictos baratos por pares o con nota | Sesgos de posición, de verbosidad y de autopreferencia |

Comprobación de contaminación: proporción de n-gramas de cada pregunta de evaluación presentes en un documento ≥ umbral → señalar y después leer.

## Comandos del lab

```text
uv run lab corpus | next-token --prompt "..." | generate [--strategy greedy|beam|sample] [--temperature T]
           [--top-k K] [--top-p P] [--repetition-penalty R] [--stop S] [--max-tokens N] [--seed N]
uv run lab decode-bench      # 2.1 diversity vs likelihood table
uv run lab chat-template     # 2.2 template and loss mask
uv run lab sft-clean         # 2.2 data audit against labels
uv run lab reward            # 2.3 reward models and the length trap
uv run lab verify | dpo      # 2.3 verifiable reward, DPO loss, KL-penalised reward
uv run lab metrics           # 2.4 EM, F1, BLEU, ROUGE-L, perplexity
uv run lab contamination     # 2.4 eval questions in the web crawl
uv run lab arena | judge     # 2.4 Elo vs Bradley–Terry, judge swap test
```

## Los tres errores más comunes

1. **Reducir el máximo de *tokens* para que deje de enrollarse.** Corta frases; arregla la condición de parada o el modelo.
2. **Fiarte de una puntuación perfecta con datos reservados.** Comprueba si los *prompts* o las respuestas se filtraron entre particiones.
3. **Preguntar a un juez LLM una sola vez, en un orden fijo.** Intercambia las posiciones, vigila la longitud y calíbralo con personas.

---

[← Anterior: Quiz](quiz.es.md) · [🏠 Módulo 02](README.es.md) · [Siguiente: Módulo 03 →](../03-llm-playground/README.es.md)
