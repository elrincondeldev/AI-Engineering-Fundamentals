🌐 [English](cheatsheet.md) · Español

# Módulo 09 — Chuleta: modelos de razonamiento

## Términos clave

| Término | Definición en una línea |
|---|---|
| Cadena de pensamiento (CoT, *chain of thought*) | Pasos intermedios escritos antes de la respuesta; cada paso pasa a ser entrada para los *tokens* siguientes |
| Modelo de razonamiento (modelo que piensa, *thinking model*) | Un modelo entrenado para razonar largo y tendido (planificar, comprobar, retroceder) antes de responder |
| *Tokens* de pensamiento (*thinking tokens*) | *Tokens* de salida gastados en razonar; se facturan y llevan tiempo aunque estén ocultos |
| Esfuerzo de razonamiento (*reasoning effort*) | Un ajuste que intercambia *tokens* de pensamiento por exactitud, con rendimientos decrecientes |
| Escalado en tiempo de inferencia | Gastar más cómputo por pregunta: muestras en paralelo o revisiones en secuencia |
| Autoconsistencia (*self-consistency*) | Muestrear varias cadenas y devolver la respuesta (normalizada) más frecuente |
| *Best-of-N* (mejor de N) | Muestrear N respuestas y quedarse con la que mejor puntúa un verificador |
| Verificador | Programa o modelo que juzga una respuesta o un paso |
| ORM / PRM | Modelo de recompensa de resultado (puntúa una solución completa) / modelo de recompensa de proceso (puntúa cada paso) |
| Árbol de pensamientos (ToT, *Tree of Thoughts*) | El modelo propone movimientos y valora estados; el código ejecuta BFS o DFS sobre ellos |
| Explotación de la recompensa (*reward hacking*) | Optimizar una puntuación indirecta premia sus sesgos en lugar del objetivo |
| STaR | Generar, conservar las justificaciones correctas, racionalizar los fallos con una pista, ajustar, repetir |
| GRPO | RL que puntúa cada muestra frente a su propio grupo; sin modelo de valor |
| Destilación | SFT de un modelo pequeño con trazas verificadas de un modelo más fuerte |
| Meta-CoT | Entrenar con trazas de búsqueda linealizadas para que la búsqueda ocurra dentro de una sola cadena |

## Fórmulas

```text
pass@k (unbiased, n samples, c correct)  = 1 - C(n - c, k) / C(n, k)
majority@k                               = accuracy of the most frequent normalised answer among k
best-of-N                                = argmax_i verifier_score(sample_i)
GRPO advantage of sample i               = (r_i - mean(r)) / std(r)      (0 for all if std = 0)
PRM chain score (this lab)               = min over steps of step_score
cost per question                        = prompt tokens + generated tokens (thinking included)
```

## Qué error, qué solución

| Tipo de error | Ejemplo | Lo detecta | No lo detecta |
|---|---|---|---|
| Desliz (aleatorio) | `6 x 7 = 52` | votación, recomprobación aritmética, autocomprobaciones | no hace falta nada especial |
| Trampa (sistemática) | contar bombillas Wi-Fi en el hub | relectura, PRM, regla específica | votación por mayoría, recomprobación aritmética |
| Cadena infiel | respuesta correcta, paso erróneo | comprobaciones de pasos, PRM | filtros que solo miran la respuesta, ORM |

## Elegir un método

| Situación | Empieza con |
|---|---|
| Pregunta corta, errores baratos | CoT normal, esfuerzo bajo |
| Pregunta larga de varios pasos | Modo con pensamiento o CoT + comprobador; sube el esfuerzo solo donde compense |
| Existe un comprobador exacto | Muestrear hasta verificar (barato, seguro, sigue mejorando con el presupuesto) |
| Sin comprobador, puedes pagar muchas muestras | *Best-of-N* con un modelo de recompensa; vigila el sesgo |
| Necesitas menos *tokens* generados y las llamadas son baratas | Búsqueda por pasos (ToT, búsqueda en haz con PRM) |
| Necesitas un modelo de razonamiento local pequeño | Destila trazas verificadas y, opcionalmente, después RL con recompensas basadas en reglas |

## Números principales del lab (razonador simulado)

| Resultado | Valor |
|---|---|
| Exactitud direct / CoT / think-high (160 problemas) | 33.1% / 49.4% / 71.9% |
| *Tokens* por respuesta, CoT frente a think-high | 72.7 frente a 194.1 |
| Votación por mayoría con N = 15 y N = 45 (110 problemas numéricos) | 89.1% y 89.1% |
| *Best-of-9* con ORM / PRM | 86.4% / 95.5% |
| Acertijos de número objetivo: CoT / muestrear hasta verificar (15) / ToT BFS con anchura 5 | 11 / 41 / 36 de 50 |
| Exactitud en test de STaR, de la ronda 0 a la ronda 4 | de 51.2% a 72.5% |
| Juguete de RL: *tokens* por respuesta, del paso 0 al paso 8 | de 77.9 a 113.0 |

## Comandos del lab

```bash
cd modules/09-reasoning-models/lab && uv sync && uv run pytest
uv run lab solve [--id r076] [--mode think-high]   # direct vs CoT vs thinking
uv run lab scale                                   # vote, best-of-N, pass@N vs tokens
uv run lab revise                                  # self-critique vs re-check vs PRM feedback
uv run lab search [--id r076]                      # verifiers, PRM beam search, reward hacking
uv run lab tot [--id r113]                         # Tree of Thoughts vs sampling until verified
uv run lab star [--no-rationalize] [--strict]      # STaR loop
uv run lab rl [--steps 8] [--group 8]              # GRPO-style toy
```

## Los tres errores más comunes

1. **Fiarse del acuerdo.** Que muchas muestras den la misma respuesta no es una prueba; los errores correlacionados (trampas) ganan las votaciones.
2. **Contar los *tokens* equivocados.** Los *tokens* de pensamiento ocultos y los *tokens* de *prompt* reenviados están los dos en la factura.
3. **Optimizar el verificador en lugar del objetivo.** Mide siempre contra una verdad de referencia que el verificador nunca vea.
