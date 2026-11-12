🌐 [English](README.md) · Español

# Lab 07 — *Workflows*, una caja de herramientas protegida y un servidor MCP para Hearthwise Assist

> Construye patrones de *workflow*, una caja de herramientas validada y con permisos sobre los datos de
> Hearthwise, y un servidor y un *host* MCP didácticos que reutilizan las mismas herramientas. ⏱️ unas
> 3 horas.

## 🎯 Qué vas a construir

| Pieza | Archivo | Qué hace |
|---|---|---|
| Escalera de autonomía | `agency.py` | Una petición resuelta en cuatro niveles de autonomía, con el coste y el riesgo de cada nivel |
| Patrones de *workflow* | `workflows.py`, `workflow_model.py` | Encadenamiento con compuertas, enrutamiento, seccionamiento, votación, reflexión, orquestador-trabajadores |
| Caja de herramientas | `tools.py`, `kb.py`, `store.py` | `get_product`, `search_help_centre`, `lookup_order`, `create_return`, cada una con un JSON Schema |
| Validador | `schema.py` | Un validador de un subconjunto de JSON Schema en Python puro |
| Ejecutor | `executor.py` | Permisos, validación, aprobación humana, límites de tiempo, reintentos, claves de idempotencia, registro de auditoría |
| Bucle de herramientas | `loop.py`, `scripted.py` | El bucle de llamada a herramientas y un modelo con guion determinista que emite objetos `ToolCall` |
| Servidor y *host* MCP | `mcp_server.py`, `mcp_client.py` | Un servidor JSON-RPC 2.0 didáctico sobre stdio y un *host* con análisis de descripciones y fijación |

Todo funciona sin conexión en una CPU con el `FakeProvider`. El almacén de pedidos es sintético y se
genera dentro del lab con una semilla fija: cinco pedidos de demostración más 55 aleatorios, todos con
los precios de las páginas de producto publicadas. El "hoy" del lab está fijado en el 2025-11-14 para
que los plazos de devolución no cambien nunca.

> El servidor y el cliente MCP de este lab son una **implementación didáctica** de un pequeño
> subconjunto del Model Context Protocol (`initialize`, `notifications/initialized`, `tools/list`,
> `tools/call`). No son el SDK oficial. Lee la especificación en https://modelcontextprotocol.io antes
> de construir uno de verdad.

## ⚙️ Preparación

```bash
cd modules/07-agents-workflows-and-tools/lab
uv sync
uv run pytest
```

Resumen esperado de los tests (el tiempo es un ejemplo y varía según la máquina):

```text
93 passed in 3.84s
```

`uv run lab --help` lista los comandos: `ladder`, `workflow`, `tools`, `mcp-demo` y `mcp-server`.

## 👣 Pasos

### Paso 1 — Instala y ejecuta los tests

Ejecuta los comandos de preparación de arriba. Todos los tests deben pasar antes de que cambies nada.
Los tests bloquean el acceso a la red; los tests de MCP arrancan el servidor como subproceso y hablan
con él por tuberías.

```bash
uv run lab --help
```

```text
usage: lab [-h] {ladder,workflow,tools,mcp-demo,mcp-server} ...

Module 07 lab: workflows, tools and MCP.

positional arguments:
  {ladder,workflow,tools,mcp-demo,mcp-server}
    ladder              one request at four levels of agency
    workflow            run a workflow pattern
    tools               list the toolbox, or run a tool-calling demo
    mcp-demo            launch the teaching MCP server and talk to it
    mcp-server          run the teaching MCP server on stdin/stdout

options:
  -h, --help            show this help message and exit
```

### Paso 2 — Sube la escalera de autonomía

```bash
uv run lab ladder
```

```text
Request: I bought an Ember T3 (order HW-310042) and changed my mind. Please start a return.

Level                         Decides     LLM calls  Tools  Tokens  Data changed
1 single LLM call             nobody              1      0      63  no
2 workflow                    your code           1      1     154  no
3 tool-using (read-only)      the model           2      1    1416  no
4 agent (can act, approved)   the model           3      2    3314  yes
```

Lee `agency.py`. Los niveles 3 y 4 llaman al mismo `_loop_rung`; la única diferencia es
`{"create_return": Permission.DENY}`. Lección: [7.1](../01-agents-overview.es.md).

### Paso 3 — Ejecuta los patrones de workflow

```bash
uv run lab workflow chain
uv run lab workflow route
uv run lab workflow section
uv run lab workflow vote
uv run lab workflow reflect
uv run lab workflow orchestrate
```

`uv run lab workflow all` los ejecuta todos de una vez. El patrón de seccionamiento imprime:

```text
== section ==
| Device | Price | Wi-Fi | Needs hub |
|---|---|---|---|
| Ember T3 | €149 | 2.4 GHz | no |
| Glow A19 | €12 | 2.4 GHz | no |
| Spark Mini | €19 | 2.4 GHz | no |
| Watcher S1 | €29 | none (HomeMesh radio) | yes |
| Lookout C2 | €89 | 2.4 GHz and 5 GHz | no |
| Hearth Hub | €69 | 2.4 GHz plus Ethernet port | no |
Sections: 6 | sequential rounds: 2 (would be 7 one after another)
LLM calls: 7 | sequential rounds: 2 | tokens: 1441
```

La lección [7.2](../02-workflow-patterns.es.md) muestra y explica la salida de cada patrón. Cada paso
de cada *workflow* es una llamada al mismo `FakeProvider`, cuyo *responder* (`workflow_model`) lee el
nombre de la tarea en la primera línea del *prompt* de sistema.

### Paso 4 — Añade una compuerta a la cadena

La cadena le pide al mensaje 5 ("Return order 310155 please.") su número de pedido, aunque el cliente
lo dio, solo que sin el prefijo `HW-`. Corrígelo **en código**, antes de la compuerta 1, en `chain_one`
dentro de `workflows.py`: si el id extraído tiene exactamente seis dígitos, ponle delante `HW-`. Después
ejecuta la cadena:

```bash
uv run lab workflow chain
```

Últimas líneas esperadas después de tu cambio:

```text
Outcomes: {'replied': 4, 'asked_for_order_id': 1, 'not_found': 1}
LLM calls: 10 | sequential rounds: 2 | tokens: 921
```

Ahora el mensaje 5 recibe "Your order HW-310155 can be returned: 19 days left. ...". Añade un test para
ello en un archivo nuevo, `tests/test_my_gate.py`. `test_chain_gate_stops_early_without_a_valid_order_id`
en `tests/test_workflows.py` y los números documentados de la cadena en
`tests/test_documented_numbers.py` fallarán ahora: actualiza el primero y deja el segundo como
recordatorio de que los números documentados están atados al código (o pon tu cambio en una rama).

<details><summary>Solución</summary>

```python
fields = json.loads(meter.ask("extract", message))
order_id = fields.get("order_id")
if order_id and re.fullmatch(r"\d{6}", order_id):
    order_id = f"HW-{order_id}"  # normalise before the gate: cheap, deterministic, testable
if not order_id or not re.match(ORDER_PATTERN, order_id):  # gate 1: stop early, ask the customer
    ...
```

```python
from m07_agents_tools import workflows as wf


def test_prefixless_order_id_is_normalised() -> None:
    result = wf.chain_one(wf.Meter(), wf.OrderStore(), "Return order 310155 please.")
    assert result["outcome"] == "replied" and "HW-310155" in result["reply"]
```

</details>

### Paso 5 — Inspecciona la caja de herramientas y el bucle de herramientas

```bash
uv run lab tools
uv run lab tools --demo fix-args
```

```text
Tool                Changes data  Required arguments                  Spec tokens
get_product         no            product                                     192
search_help_centre  no            query                                       194
lookup_order        no            order_id                                    150
create_return       yes           order_id, sku, quantity, reason             331
All four specs together: 872 tokens, sent with every LLM call.
== fix-args ==
Request: Where is my order 310108?
  LLM call 1 -> lookup_order({"order_id": "310108"})
      invalid_arguments $.order_id: '310108' does not match pattern ^HW-\d{6}$. Fix the arguments and call the tool a...
  LLM call 2 -> lookup_order({"order_id": "HW-310108"})
      ok (1 attempt)
  LLM call 3 -> answer: Order HW-310108 is shipped.
  Totals: 3 LLM calls, 2 tool calls, 3073 prompt tokens, 29 completion tokens
  Returns in the store: 0
```

Abre `scripted.py` y encuentra la línea que hace que el modelo copie lo que escribió el cliente, y la
línea que lo corrige después de leer el error. Después lee `run_tool_loop` en `loop.py`: todo el bucle
en una sola función corta.
Lección: [7.3](../03-tool-calling.es.md).

### Paso 6 — Haz segura la ejecución de herramientas

Ejecuta las demos que ponen a prueba la aprobación, la propiedad de los datos, los reintentos, los
límites de tiempo y la idempotencia:

```bash
uv run lab tools --demo return
uv run lab tools --demo return-denied
uv run lab tools --demo not-yours
uv run lab tools --demo retries
uv run lab tools --demo timeout
uv run lab tools --demo idempotency
```

La demo `retries` inyecta dos fallos transitorios; la demo `timeout` hace que una herramienta duerma
0,5 s frente a un límite de tiempo de 0,2 s. Líneas clave esperadas:

```text
== retries ==
Request: Where is order HW-310108?
  LLM call 1 -> lookup_order({"order_id": "HW-310108"})
      ok (3 attempts)
```

```text
== not-yours ==
Request: Where is order HW-418230?
  LLM call 1 -> lookup_order({"order_id": "HW-418230"})
      error No order HW-418230 found for this customer.
  LLM call 2 -> answer: I could not find order HW-418230 in your account. Could you check the number?
  Totals: 2 LLM calls, 1 tool call, 1914 prompt tokens, 30 completion tokens
  Returns in the store: 0
```

En la demo `idempotency`, compara `Returns in the store: 1` (claves activadas) con
`Returns in the store: 2` (claves desactivadas). Después lee `ToolExecutor._execute` y
`_run_with_retries` en `executor.py` y responde: ¿en qué orden se hacen las comprobaciones y qué
errores no se reintentan nunca?

### Paso 7 — Añade una quinta herramienta

Añade una herramienta de solo lectura, `check_warranty`, que devuelva, para un pedido entregado de
quien ha iniciado sesión, la fecha de fin de la garantía estándar de 2 años y la de la garantía de 3
años con Hearthwise Plus (consulta `kb/policies/warranty.md`). Necesitas tres cosas: un método
`warranty()` en `OrderStore`, una entrada `Tool` en `build_toolbox` con una descripción y un JSON
Schema, y nada más. Después compruébala de tres maneras:

```bash
uv run lab tools          # a fifth row appears
uv run lab mcp-demo       # the server now lists five tools, with no protocol code changed
uv run python -c "
from m07_agents_tools.executor import ToolExecutor
from m07_agents_tools.providers import ToolCall
from m07_agents_tools.tools import Session, build_toolbox
ex = ToolExecutor(build_toolbox(), Session('C-0042'))
print(ex.execute(ToolCall('c1', 'check_warranty', {'order_id': 'HW-310042'})).data)
"
```

Salida esperada del último comando:

```text
{'order_id': 'HW-310042', 'delivered_on': '2025-10-23', 'standard_until': '2027-10-23', 'with_plus_until': '2028-10-23'}
```

Los tests que fijan la caja de herramientas en cuatro herramientas (en `test_mcp.py`, `test_cli.py` y
`test_documented_numbers.py`) fallarán. Es el contrato de los números documentados funcionando: las
lecciones citan cuatro herramientas y 872 *tokens*. Mantén tu herramienta en una rama, o actualiza los
tests y los números a la vez. En este lab, el *host* calcula sus fijaciones aprobadas a partir del
mismo código, así que confía en tu herramienta nueva de inmediato; en producción, las fijaciones salen
de la última revisión, y `check_warranty` quedaría en cuarentena hasta que Noor la aprobara.

<details><summary>Solución</summary>

```python
# store.py, inside OrderStore
def warranty(self, order_id: str, customer_id: str) -> dict[str, object]:
    order = self._owned(order_id, customer_id)
    if order.delivered_on is None:
        raise ToolError(f"Order {order_id} has not been delivered yet.")
    d = order.delivered_on
    return {"order_id": order_id, "delivered_on": d.isoformat(),
            "standard_until": d.replace(year=d.year + 2).isoformat(),
            "with_plus_until": d.replace(year=d.year + 3).isoformat()}
```

```python
# tools.py, add to the list in build_toolbox
Tool(
    name="check_warranty",
    description=(
        "Get the warranty end dates of one delivered order of the signed-in customer: the standard "
        "2-year warranty and the 3-year warranty with Hearthwise Plus. Use it when a customer reports "
        "a faulty device. Read-only."
    ),
    parameters=_obj({"order_id": ORDER_ID}, ["order_id"]),
    handler=lambda a, s: store.warranty(a["order_id"], s.customer_id),
    timeout_s=timeout_s,
),
```

</details>

### Paso 8 — Habla MCP a mano

Envía tú mismo mensajes JSON-RPC al servidor didáctico, uno por línea:

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"teaching-subset","capabilities":{},"clientInfo":{"name":"terminal","version":"0"}}}' \
  '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
  'this is not json' | uv run lab mcp-server
```

```text
{"jsonrpc": "2.0", "id": 1, "error": {"code": -32600, "message": "Server not initialized: send initialize first"}}
{"jsonrpc": "2.0", "id": 2, "result": {"protocolVersion": "teaching-subset", "capabilities": {"tools": {}}, "serverInfo": {"name": "hearthwise-tools", "version": "0.1.0"}}}
{"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": "Parse error"}}
hearthwise-tools: stdin closed, exiting
```

Entran cuatro mensajes y salen tres respuestas: la notificación no recibe ninguna. Ahora lee
`TeachingMcpServer.handle` en `mcp_server.py` y encuentra de dónde salió cada respuesta. Lección:
[7.4](../04-model-context-protocol.es.md).

### Paso 9 — Ejecuta el host y detecta un servidor envenenado

```bash
uv run lab mcp-demo
uv run lab mcp-demo --poisoned
```

La primera ejecución muestra el saludo inicial, la revisión del *host*, dos llamadas directas (una
rechazada porque el pedido pertenece a otro cliente) y el bucle de herramientas de la lección 7.3
funcionando sobre MCP, y termina con `JSON-RPC messages exchanged: 13`. La segunda ejecución se conecta
a un servidor con una herramienta `check_stock` envenenada:

```text
  check_stock          findings: ['hidden-instruction markup', 'tells the model to hide something', 'mentions other tools: lookup_order']
  pin diff: {'added': ['check_stock'], 'removed': [], 'changed': []}
  exposed to the model: ['get_product', 'search_help_centre', 'lookup_order', 'create_return'] | quarantined: ['check_stock']
```

Lee `remote_tools`, `lint_description`, `pin_tools` y `diff_pins` en `mcp_client.py`. ¿Cuáles de ellas
te seguirían protegiendo si la herramienta envenenada hubiera estado ahí desde la primera revisión?

## 💥 Rómpelo

**1. Quita el patrón del id de pedido.** Borra `"pattern"` de `ORDER_ID` en `tools.py` y ejecuta
`uv run lab tools --demo fix-args`. ¿Qué le dicen al cliente sobre un pedido que existe?

<details><summary>Qué pasa</summary>

El `"310108"` del modelo pasa ahora la validación, el almacén no encuentra ese pedido y la herramienta
devuelve un error de negocio en lugar de `invalid_arguments`. El modelo no puede saber que el argumento
estaba mal formado, así que le dice al cliente "I could not find order HW-310108 in your account" ("No
encuentro el pedido HW-310108 en tu cuenta") sobre un pedido que está en camino. Un esquema preciso
convierte una respuesta engañosa en una autocorrección.

</details>

**2. Desactiva las claves de idempotencia.** En `run_demo`, dentro de `cli.py`, cambia el valor por
defecto `keys: bool = True` a `False` y ejecuta `uv run lab tools --demo idempotency`.

<details><summary>Qué pasa</summary>

Las dos mitades imprimen ahora `Returns in the store: 2`. Al cliente se le dijo "Done" una vez y tiene
dos devoluciones abiertas, lo que significa dos reembolsos. El *timeout* ocultó que la primera llamada
ya había funcionado.

</details>

**3. Deja al bucle sin pasos.** Llama a `run_tool_loop(..., max_steps=1)` en `run_demo` y ejecuta la
demo `return`.

<details><summary>Qué pasa</summary>

La primera llamada del modelo es `lookup_order`; después, el bucle se detiene con
`stop_reason == "max_steps"` y una respuesta vacía. Los presupuestos te protegen de los bucles
descontrolados, pero un presupuesto demasiado pequeño hace fallar cualquier petición legítima de varios
pasos. Registra el motivo de parada y genera una alerta con `max_steps`.

</details>

**4. Deja pasar el HTML oculto.** En `_TextExtractor.handle_starttag`, dentro de `kb.py`, haz que
`hidden` sea siempre `False` y después ejecuta
`uv run python -c "from m07_agents_tools import kb; print(kb.search('refund verify')[0]['id'])"`.

<details><summary>Qué pasa</summary>

Imprime `kb-025`: la página de la comunidad pasa al primer puesto, y su fragmento es el párrafo oculto
dirigido a los asistentes de IA. La salida de una herramienta de búsqueda va directa al contexto del
modelo, así que todo lo que esconda una página pasa a formar parte del *prompt*. Descartar los
elementos ocultos es una capa; tratar los resultados como datos es la otra.

</details>

**5. Fíate de lo que dice el servidor.** En `remote_tools`, dentro de `mcp_client.py`, pon
`side_effect=False` en todas las herramientas y ejecuta `uv run lab mcp-demo`.

<details><summary>Qué pasa</summary>

La línea `[approval]` desaparece: `create_return` se ejecuta sobre MCP sin que ninguna persona lo
apruebe. El *host* acaba de dejar que un servidor remoto decida qué cuenta como inofensivo. Mantén la
lista de herramientas de escritura en el *host*.

</details>

## 🚀 Amplíalo

**1. Limita el orquestador.** Añade `max_rounds: int = 3` a `orchestrate` y deja de añadir
trabajadores cuando se alcance; registra `"capped": True` en el informe.

<details><summary>Pista</summary>

Cuenta las iteraciones del bucle `while todo:`. Cuando se alcance el límite, sal del bucle antes de
volver a llamar a `replan` y ejecuta igualmente `synthesise` con lo que tengas. Pruébalo con un
`replan` falso que siempre pida un producto más.

</details>

**2. No reintentes efectos secundarios ambiguos.** Cuando una herramienta con efectos secundarios
agote su tiempo de espera y no lleve una clave de idempotencia, devuelve un resultado `timeout` con el
mensaje "Outcome unknown: check the order before trying again" en lugar de reintentar.

<details><summary>Pista</summary>

En `_run_with_retries`, pasa como parámetro si hay una clave. Ante un `FutureTimeout`, si
`tool.side_effect and not has_key`, devuelve el resultado de inmediato. Con el Rómpelo 2 aplicado, la
demo de idempotencia debería dejar ahora una sola devolución en el almacén y decirle al cliente que el
resultado es desconocido.

</details>

**3. Añade recursos al servidor didáctico.** Ofrece la política de devoluciones vigente (kb-011) como
recurso MCP con `resources/list` y `resources/read`, y anuncia la capacidad en `initialize`.

<details><summary>Pista</summary>

Consulta la forma exacta de las peticiones y los resultados en la sección de recursos de la
especificación, en https://modelcontextprotocol.io, antes de escribir código. Devuelve el texto del
documento, no el HTML. Limita `resources/read` a documentos públicos y vigentes; el manual interno no
debe aparecer nunca en la lista.

</details>

**4. Dale al bucle un presupuesto de *tokens*.** Añade `max_prompt_tokens` a `run_tool_loop`. Cuando una
llamada al modelo pida herramientas y el total acumulado de *tokens* del *prompt* ya supere el
presupuesto, detén el bucle con `stop_reason = "budget"` *antes* de ejecutar esas herramientas.

<details><summary>Pista</summary>

La respuesta del modelo lleva `prompt_tokens`; súmalo después de cada llamada. En la demo `compare`,
las dos llamadas cuestan 932 y 1.343 *tokens* de *prompt*; en la demo `return`, las tres llamadas
cuestan 937, 1.079 y 1.198. Con un presupuesto de 2.000, `compare` sigue respondiendo, mientras que
`return` se detiene con un total acumulado de 2.016, antes de que se ejecute `create_return`. Comprobar
antes de ejecutar las herramientas significa que un presupuesto nunca puede interrumpir un efecto
secundario a medias.

</details>

**5. Guarda las fijaciones.** Guarda las fijaciones aprobadas en `lab/outputs/approved_pins.json` la
primera vez que se ejecute `mcp-demo`, y cárgalas en las ejecuciones siguientes, para que una
herramienta que añadas en el paso 7 quede en cuarentena hasta que borres el archivo.

<details><summary>Pista</summary>

Usa `pathlib.Path("outputs")`, `mkdir(exist_ok=True)` y `json.dumps(pins, indent=2)`. Carga el archivo
si existe; si no, calcula las fijaciones a partir del `tools/list` en vivo y escríbelas. Git ignora
`outputs/`.

</details>

## ✅ Definición de terminado (*Definition of Done*)

- [ ] `uv run pytest` pasa en una copia limpia del repositorio.
- [ ] Ejecutaste `ladder`, los seis patrones de `workflow`, cada `tools --demo` y las dos variantes de
      `mcp-demo`.
- [ ] Paso 4: los ids de pedido sin prefijo se normalizan antes de la compuerta 1, con un test.
- [ ] Paso 7: `check_warranty` existe, valida su entrada, usa el id de cliente de la sesión y aparece
      por MCP sin cambios en el protocolo.
- [ ] Sabes explicar cada estado del ejecutor (`invalid_arguments`, `denied`, `rejected`, `timeout`,
      `error`) y si el modelo debería reintentarlo.
- [ ] Hiciste al menos tres experimentos de Rómpelo y escribiste una frase sobre cada uno.
- [ ] Completaste al menos un elemento de Amplíalo, con un test.

## 🧯 Solución de problemas

| Síntoma | Solución |
|---|---|
| `uv: command not found` | Instala uv desde https://docs.astral.sh/uv/ y abre una terminal nueva |
| `No interpreter found for Python 3.12` | `uv python install 3.12` y después otra vez `uv sync` |
| `Could not find the repo's datasets/ folder` | Ejecuta desde dentro del repositorio o define `COURSE_DATASETS_DIR` con la ruta de `datasets` |
| `McpError: initialize: no response within 10.0 s` | El servidor falló al arrancar. Ejecuta `uv run lab mcp-server < /dev/null` para ver su error en la salida de error estándar |
| Los tests fallan después del paso 4 o del paso 7 | Es lo esperado: algunos tests fijan los números documentados. Consulta las notas de esos pasos |
| `error: Cannot reach http://localhost:11434/api/chat` | Usaste `--live` con `LLM_PROVIDER=ollama` y Ollama no está en marcha |

## 🔌 Usar un modelo real

Todos los comandos anteriores usan el modelo con guion para que los resultados sean reproducibles. El
propio bucle de herramientas no depende del proveedor: `run_tool_loop` envía `executor.specs()`
(definiciones de funciones al estilo de OpenAI) a través de la capa de proveedores compartida de
`providers.py`, que serializa las llamadas a herramientas y sus resultados para Ollama y para los
*endpoints* compatibles con OpenAI.

```bash
# Ollama: start the server, pull a model that supports tool calling, then:
LLM_PROVIDER=ollama OLLAMA_MODEL=<a tool-capable model> uv run lab tools --demo compare --live

# Any OpenAI-compatible endpoint:
LLM_PROVIDER=openai OPENAI_BASE_URL=<url> OPENAI_API_KEY=<key> OPENAI_MODEL=<model> \
  uv run lab tools --demo compare --live
```

Con `--live`, las demos usan `get_provider()` en lugar del modelo con guion. Espera trazas distintas:
un modelo real puede elegir otras herramientas, formular los argumentos de otra manera o responder sin
herramientas. Las garantías del ejecutor (validación, permisos, aprobación, idempotencia) se mantienen
haga lo que haga el modelo, y para eso precisamente se ponen en el código.

**Qué no se probó:** la ruta `--live` no se ejecutó contra un modelo real mientras se escribía este
lab. Ollama estaba instalado, pero no en marcha, en el entorno de autoría, y no se usó ninguna clave de
API. Los tests solo comprueban, sin conexión, que la capa de proveedores construye correctamente el
contenido de las peticiones y que `--live` informa de un error claro cuando no se puede llegar a ningún
servidor. Los tests en vivo se marcarían con `@pytest.mark.ollama` o `@pytest.mark.openai` y se
ejecutarían con `RUN_LIVE_TESTS=1`.

---

[← Anterior: 7.4 El Model Context Protocol](../04-model-context-protocol.es.md) · [🏠 Módulo 07](../README.es.md) · [Siguiente: Quiz →](../quiz.es.md)
