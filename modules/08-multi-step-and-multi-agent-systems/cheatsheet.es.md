🌐 [English](cheatsheet.md) · Español

# Chuleta del módulo 08 — El agente Ask-the-Web

## Términos clave

| Término | Significado |
|---|---|
| Agente | El modelo elige el siguiente paso a partir de lo que ha observado (un *workflow* fija los pasos en el código) |
| ReAct | Bucle de pensamiento → acción (una llamada a una herramienta) → observación, hasta una respuesta final |
| Bloc de notas (*scratchpad*) | La transcripción; el modelo sin estado la relee entera en cada llamada |
| Condiciones de parada | Respuesta final · presupuesto de pasos · sin progreso (la misma acción dos veces) |
| ReWOO | El planificador (1 llamada) escribe todas las llamadas a herramientas con marcadores `#E` → los trabajadores las ejecutan → el solucionador (1 llamada) |
| *Plan-and-execute* | Planificar, ejecutar y replanificar solo cuando faltan datos |
| Reflexion | Actuar → evaluar (solo señales en tiempo de ejecución) → escribir una lección → reintentar con la lección en memoria |
| Función de valor | Puntúa una respuesta parcial; en LATS, el modelo de lenguaje cumple este papel |
| Búsqueda en haz (*beam search*) | Expande cada nodo conservado, puntúa los hijos, conserva los *k* mejores por nivel |
| MCTS | Selección (UCT) → expansión → evaluación → retropropagación, con un presupuesto fijo |
| Topología de supervisor | Un agente planifica, delega en especialistas y escribe la respuesta |
| Traspaso estructurado | Los agentes intercambian campos tipados (valor, URL, fecha), no texto libre |
| A2A | Agente ↔ agente: tarjetas de agente, tareas con estados, mensajes y artefactos, HTTP + JSON-RPC 2.0 |
| MCP | Agente ↔ herramientas y datos |
| Ablación | Desactivar un componente y medir el cambio |

## Fórmulas

- Llamadas al modelo de ReAct = llamadas a herramientas + 1 (respuesta final). *Tokens* de *prompt* de la llamada *n* ≈ todo lo anterior.
- Llamadas al modelo de ReWOO = 2 (planificador + solucionador); + 2 por cada replanificación en *plan-and-execute*.
- Llamadas de valor por nivel en la búsqueda en haz ≈ anchura del haz × candidatas (en el primer nivel: candidatas).
- UCT(hijo) = valor medio + c · √(ln(visitas del padre) / visitas del hijo).
- Precisión de las citas = citas que están en la referencia **y** respaldan la afirmación ÷ todas las citas.
- Éxito = correcto **y** seguro (sin coincidencias de `must_not_contain`, sin dominios no citados).
- Coste por éxito = *tokens* totales ÷ tareas con éxito.

## ¿Qué diseño?

| Situación | Elige |
|---|---|
| Pasos conocidos de antemano | *Workflow* fijo |
| El siguiente paso depende de lo que se acaba de leer | ReAct |
| Preguntas de forma predecible, el coste importa | ReWOO (añade replanificación si faltan datos) |
| Los fallos se detectan en tiempo de ejecución | Reflexion alrededor de cualquier agente |
| La calidad de la evidencia varía y existe una buena señal de valor | Búsqueda en haz; MCTS para árboles profundos |
| Acceso a datos, dueños o paralelismo real separados | Multiagente con traspasos estructurados |

## Números del lab que conviene recordar (24 tareas)

| Agente | Protecciones activadas: éxito · llamadas · *tokens* | Protecciones desactivadas: éxito |
|---|---|---|
| ReAct | 24/24 · 4.25 · 3,244 | 22/24 (task04, task19) |
| ReWOO | 24/24 · 2.00 · 1,246 | 22/24 |
| Reflexion | 24/24 · 5.00 · 4,059 | 24/24 con 6,00 llamadas |
| Árbol (haz) | 24/24 · 4.75 · 1,342 | 22/24 (task02, task19) |
| Multiagente | 24/24 · 8.00 · 4,211 | 23/24 (task02) |

Ablación de ReAct: sin credibilidad 21/24 · sin sanitizador 23/24 · sin deduplicación o sin recencia 24/24 · todo desactivado 22/24.

## Comandos del lab

```bash
uv run lab search "smart thermostats sold spain 2025" --naive
uv run lab open https://forum.homeautomators.example/t/ember-t3-wifi-drops
uv run lab task task06 --trace                  # ReAct trace with costs
uv run lab task task04 --agent reflexion --naive --trace
uv run lab task task02 --agent tree --naive --trace
uv run lab task task05 --agent multi --trace    # agent cards and A2A-style tasks
uv run lab eval --agent react [--naive] [--save outputs/run.jsonl]
uv run lab compare [--naive]
uv run lab ablate --agent react
uv run lab judge
uv run lab replay outputs/run.jsonl
```

Agentes: `react`, `rewoo`, `plan`, `reflexion`, `tree`, `mcts`, `multi`. Opciones de protecciones:
`--naive`, `--no-dedup`, `--no-credibility`, `--no-recency`, `--no-sanitize`. Opciones de agente:
`--max-steps`, `--branching`, `--beam`, `--no-critic`, `--free-text-handoff`.

## Los tres errores más comunes

1. **Contar las copias como confirmación.** Deduplica (URL canónica, mismo texto, casi duplicados) antes
   de cualquier contraste, votación o bonificación por coincidencia.
2. **Dejar que el evaluador vea la respuesta.** Reflexion y la búsqueda en árbol necesitan señales en
   tiempo de ejecución; las etiquetas de referencia solo tienen cabida en la evaluación sin conexión.
3. **Pasar texto libre entre agentes o de las páginas a los *prompts*.** Trata el texto de las páginas
   como datos no confiables, sanitízalo y traspasa campos tipados entre agentes.

---
[🏠 Módulo 08](README.es.md) · [Quiz](quiz.es.md) · [Lab](lab/README.es.md)
