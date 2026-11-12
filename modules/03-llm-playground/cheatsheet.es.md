🌐 [English](cheatsheet.md) · Español

# Chuleta del módulo 03 — Construye un playground de LLM

## Componentes: una tarea cada uno

| Componente | Tarea | Archivo del lab |
|---|---|---|
| Cliente | Entrada y salida | `cli.py` |
| Capa de API | Autenticación, límites de tasa, cuotas | no se construye |
| Orquestador | Ejecuta las etapas en orden | `playground.py` |
| Barreras de entrada | Anonimizar datos personales (PII), marcar inyecciones, frontera de datos | `redact.py`, `guardrails.py` |
| Ensamblado del *prompt* | Sistema + historial + mensaje dentro del presupuesto | `conversation.py` |
| Pasarela de modelos | Una interfaz; *timeouts*, reintentos, respaldo | `resilience.py`, `providers.py` |
| Barreras de salida | `empty`, `truncated`, `pii_in_output`, afirmaciones prohibidas | `guardrails.py` |
| Almacén de conversaciones | Historial por sesión (anonimizado) | `Conversation` |
| Trazas | Una línea JSON por petición, sin texto en bruto | `tracing.py` |
| Bucle de evaluación | Trazas → casos de test → cambios con control | `compare.py` |

## Fórmulas

| Qué | Fórmula |
|---|---|
| Tiempo hasta el primer *token* (TTFT) | sobrecarga + prellenado por *token* × *tokens* del *prompt* |
| Latencia total | TTFT + decodificación por *token* × (*tokens* de respuesta − 1) |
| Velocidad de decodificación | (n − 1) ÷ (total − TTFT) |
| Rendimiento de extremo a extremo | n ÷ total |
| Coste por petición | (*prompt* × precio de entrada + respuesta × precio de salida) ÷ 1.000.000 |
| Peticiones al mes | personas usuarias × peticiones por persona y día × días laborables |
| Presupuesto de entrada | ventana de contexto − `max_tokens` |
| *Prompt* en el turno n | ≈ S + (n − 1) × k + mensaje nuevo |
| Total en N turnos | ≈ N × (S + mensaje) + k × N × (N − 1) ÷ 2 (cuadrático) |
| Éxito con r reintentos | 1 − p^(r + 1) para fallos independientes con probabilidad p |
| Techo de la espera | min(máximo, base × 2^(reintento − 1)); con variación aleatoria completa espera uniform(0, techo) |

Perfiles simulados (inventados): `fake-small` 120 ms de sobrecarga, 0,25 ms/*token* de *prompt*,
20 ms/*token*, ventana de 1.024; `fake-large` 300 ms, 0,60 ms, 45 ms, ventana de 8.192. Precios de
ejemplo por millón de *tokens*: 0,10 €/0,40 € y 1,00 €/4,00 €.

## Tipos de error

| Tipo | Reintentar | Recurrir al respaldo |
|---|---|---|
| `timeout`, `rate_limit` (429), `server` (5xx), `network` | sí, espera exponencial + variación aleatoria | sí |
| `auth` (401/403) | no | sí |
| `bad_request` (400), `context_length` | no: arregla la petición | no |

Dimensiona los *timeouts* para que cada intento más su espera quepan en el plazo total, con margen para
el respaldo.

## Políticas de truncado (grabación de 20 turnos, presupuesto de 768)

| Política | Total enviado | Por encima del presupuesto | Sin *prompt* de sistema | Recuerda el pedido |
|---|---|---|---|---|
| `keep_all` | 12.657 | 8 turnos | 0 | sí |
| `sliding_window` (6 mensajes) | 3.110 | 0 | 17 turnos | no |
| `pin_system` | 11.046 | 0 | 0 | no |
| `summarise` | 11.128 | 0 | 0 | sí |

## Niveles del anonimizador (1.224 tickets: 1.224 correos, 82 teléfonos, 82 nombres)

| Nivel | Nombres detectados | Falsos positivos | Precisión | Tickets con fugas |
|---|---|---|---|---|
| `naive` | 0/82 | 200 (fechas) | 0.867 | 82 |
| `regex` | 0/82 | 0 | 1.000 | 82 |
| `names` | 75/82 | 0 | 1.000 | 7 (mayúsculas) |

## Comandos del lab

```bash
uv run lab ask "How long do I have to return a Glow A19?" [--stream] [--flaky server,rate_limit]
uv run lab chat                 # /system /temp /model /max /reset /history /cost /quit
uv run lab retry                # scripted failures + 1,000-request reliability table
uv run lab budget [--budget 400] [--show-summary]
uv run lab redact [--ticket T00008]
uv run lab guard                # injection tripwires + data boundary
uv run lab compare [--show]     # models x temperatures x 10 seeds
uv run lab cost [--users 60 --requests-per-user 25 --days 21]
LLM_PROVIDER=ollama OLLAMA_MODEL=llama3.2:1b uv run lab ask "..."   # local model (untested here)
```

## Los tres errores más comunes

1. **Barreras de protección en el cliente.** Impón la anonimización y la frontera de datos en el
   orquestador; falla en cerrado con los proveedores desconocidos.
2. **Reintentarlo todo, sin variación aleatoria ni plazo total.** Clasifica primero; falla rápido ante
   las peticiones incorrectas; añade un disyuntor (*circuit breaker*) para las caídas.
3. **Una ventana deslizante que descarta el *prompt* de sistema, y un presupuesto que se olvida de
   `max_tokens`.** Fija el *prompt* de sistema, presupuesta por *tokens* y mantén juntos los pares
   `user`/`assistant`.

---

[🏠 Módulo 03](README.es.md) · [Quiz](quiz.es.md) · [Lab](lab/README.es.md)
