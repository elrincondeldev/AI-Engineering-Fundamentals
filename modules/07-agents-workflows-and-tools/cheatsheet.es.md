🌐 [English](cheatsheet.md) · Español

# Chuleta del módulo 07 — *Workflows*, herramientas y MCP

## Definiciones clave

| Término | En una línea |
|---|---|
| Llamada al LLM | Un *prompt* de entrada, una respuesta de salida; nadie decide un siguiente paso |
| *Workflow* | Tu código define los pasos; las llamadas al LLM los completan |
| Agente | El modelo decide el siguiente paso (qué herramienta, cuándo parar) dentro de los límites que impone tu código |
| Sistema agéntico | Término general para *workflows* y agentes |
| Llamada a herramienta | Una petición estructurada del modelo (`id`, `name`, `arguments` en JSON); tu código la ejecuta |
| Clave de idempotencia | Una clave que hace inofensivo repetir una acción: misma clave, mismo resultado, sin segunda acción |
| MCP | Model Context Protocol: un estándar para que las aplicaciones de IA descubran y llamen herramientas en servidores |
| *Host* / cliente / servidor | La app que usa la persona / el conector dentro de ella, uno por servidor / el programa que ofrece herramientas |

## La escalera de autonomía (lab: `uv run lab ladder`)

| Nivel | Quién decide | Llamadas al LLM | *Tokens* | Datos modificados |
|---|---|---|---|---|
| 1 Una sola llamada al LLM | nadie | 1 | 63 | no |
| 2 *Workflow* | tu código | 1 | 154 | no |
| 3 Con herramientas (solo lectura) | el modelo | 2 | 1.416 | no |
| 4 Agente (puede actuar, con aprobación) | el modelo | 3 | 3.314 | sí |
| 5 Multiagente | varios modelos | módulo 08 | | |

**No** construyas un agente cuando los pasos se conocen, ya existe un botón, una acción equivocada es
costosa de deshacer, no puedes evaluarlo o el presupuesto de latencia es ajustado.

## Patrones de workflow (lab: `uv run lab workflow <pattern>`)

| Patrón | Úsalo cuando | Resultado en el lab |
|---|---|---|
| Encadenamiento + compuertas | Pasos conocidos y comprobables | 9 llamadas al LLM para 6 mensajes en lugar de 12 |
| Enrutamiento | Tipos de entrada distintos | 0.848 de exactitud, 0.954 cuando enruta, 136 de 1.224 a personas |
| Seccionamiento | Partes independientes | 7 llamadas, 2 rondas secuenciales en lugar de 7 |
| Votación | Despistes aleatorios, mucho en juego | k=1/3/5: 0.613 / 0.697 / 0.770; techo 0.848 |
| Reflexión | Hay pruebas o una rúbrica | 3 problemas en la ronda 1, aprobado en la ronda 2 |
| Orquestador-trabajadores | Las subtareas dependen de lo que se encuentra | Se añade el hub después de que un trabajador informe `needs_hub` |

La latencia sigue a las **rondas secuenciales**; el coste sigue a las **llamadas al LLM y los *tokens***.

## Una buena definición de herramienta

- Nombre: verbo + sustantivo (`lookup_order`).
- Descripción: qué hace, cuándo usarla, cuándo no, qué devuelve, unidades, "cambia datos".
- Esquema: `type`, `enum` para listas cerradas, `pattern` para formatos, `minimum`/`maximum`, `required`,
  `"additionalProperties": false`.
- Nunca es un parámetro: la identidad de quien lo usa. Tómala de la sesión.
- Cada definición se envía con cada llamada: las cuatro del lab cuestan 872 *tokens*.

## El orden de comprobaciones del ejecutor

```text
exists? -> permission (allow / approve / deny) -> validate schema -> human approval (side effects)
       -> run with timeout -> retry transient errors with backoff (same idempotency key) -> audit log
```

| Estado | El modelo debería... |
|---|---|
| `invalid_arguments` | corregir los argumentos y volver a llamar |
| `unknown_tool` | elegir una herramienta de la lista |
| `denied` / `rejected` | no reintentar; explicarlo o pasar el caso |
| `timeout` / `error` | normalmente no reintentar; explicarlo o escalar |

## MCP en una pantalla

```text
-> {"jsonrpc":"2.0","id":1,"method":"initialize","params":{protocolVersion, capabilities, clientInfo}}
<- {"jsonrpc":"2.0","id":1,"result":{protocolVersion, capabilities, serverInfo}}
-> {"jsonrpc":"2.0","method":"notifications/initialized"}            (notification: no id, no reply)
-> {"jsonrpc":"2.0","id":2,"method":"tools/list"}
<- {"jsonrpc":"2.0","id":2,"result":{"tools":[{name, description, inputSchema}]}}
-> {"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":..., "arguments":{...}}}
<- {"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":...}],"isError":false}}
```

- Primitivas: **herramientas** (decide el modelo), **recursos** (decide la aplicación), ***prompts*** (decide la persona).
- Transportes: **stdio** (subproceso, un mensaje JSON por línea, *logs* en stderr) y basado en HTTP (remoto).
- Problemas de protocolo: errores JSON-RPC (-32700 análisis, -32600 petición no válida, -32601 método no
  encontrado, -32602 parámetros no válidos). Fallos de herramienta: `result` con `isError: true`.

## Seguridad en MCP

| Riesgo | Mitigación |
|---|---|
| Envenenamiento de herramientas (instrucciones en las descripciones) | Analizar las descripciones con un *linter*, revisar, fijar las definiciones, poner en cuarentena las herramientas nuevas o modificadas |
| Permisos demasiado amplios | Credenciales acotadas por servidor; denegar las herramientas de escritura en los *hosts* de solo lectura |
| Delegado confundido (*confused deputy*) | Identidad tomada de la sesión o del entorno de arranque, y el servidor la hace cumplir |
| Servidores no confiables | Revisar el código, fijar las versiones, entorno mínimo, tratar los resultados como datos |

## Comandos del lab

```bash
uv run lab ladder                      # one request, four levels of agency
uv run lab workflow all                # chain, route, section, vote, reflect, orchestrate
uv run lab tools                       # tool table and spec token costs
uv run lab tools --demo all            # fix-args, compare, approval, retries, timeout, idempotency
uv run lab mcp-demo [--poisoned]       # host + teaching MCP server over stdio
uv run lab mcp-server                  # the server alone, on stdin/stdout
```

## Los tres errores más comunes

1. **Reglas de seguridad solo en el *prompt*.** Haz cumplir los permisos, la aprobación y la identidad
   en el ejecutor.
2. **Reintentar efectos secundarios a ciegas.** Un *timeout* no dice si la acción ocurrió; usa claves
   de idempotencia o informa de "resultado desconocido".
3. **Fiarse del texto de herramientas y servidores.** Las descripciones y los resultados son datos
   escritos por otras personas; pásalos por un *linter*, fíjalos y no sigas nunca las instrucciones que
   contengan.

---

[🏠 Módulo 07](README.es.md) · [Quiz](quiz.es.md) · [Lab](lab/README.es.md)
