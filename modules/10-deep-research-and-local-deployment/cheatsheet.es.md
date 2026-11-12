🌐 [English](cheatsheet.md) · Español

# Chuleta del módulo 10 — *Deep research* y despliegue local

## El bucle de investigación

| Etapa | Contrato | Quién lo hace mejor |
|---|---|---|
| Planificar | Pregunta → 2–6 subconsultas con aspecto y entidades (JSON validado; alternativa = la propia pregunta) | Modelo de razonamiento |
| Buscar | Subconsulta → las k mejores páginas (BM25 en el lab, k = 4) | Buscador |
| Filtrar | Página → credibilidad 0–1 con motivos; descartar por debajo de 0,4 | Código |
| Leer | Página sanitizada → citas textuales; conservar solo las citas encontradas en la página | Modelo pequeño + código |
| Notas | Pasaje → nota con URL, medio, fecha, credibilidad, actualidad y relevancia por subconsulta | Código |
| Reflexionar | Notas por subconsulta → entidades que faltan → búsquedas de seguimiento | Código o modelo |
| Redactar | Notas numeradas + lagunas → informe con citas `[n]`, lagunas y salvedades | Modelo de razonamiento |
| Calificar | Informe → hechos cubiertos y respaldados, citas válidas, dominios, baja credibilidad, trampas, PII | Código |

**Parar cuando:** todas las subconsultas están cubiertas · una iteración no aporta evidencia nueva · se agota el presupuesto (iteraciones, búsquedas, páginas, *tokens*).

## Cinco modos de fallo

| Fallo | Defensa |
|---|---|
| Cita alucinada | Eliminar los marcadores sin fuente; comprobar cada número citado contra la página citada |
| Fuente poco creíble | Política de fuentes + señales de contenido (relleno, gritos, lenguaje de ventas, publicaciones de foro) |
| Instrucciones inyectadas | Sanitizar antes de que ningún modelo lea; verificar las citas después; herramientas mínimas para el agente |
| Datos desactualizados | Peso por actualidad (vida media de 365 días); gana la página más reciente del mismo medio |
| Parada prematura | Parada basada en la cobertura; informar de los dominios distintos junto a los hechos |

## Fórmulas clave

- **Puntuación de una nota** = relevancia × credibilidad × actualidad; actualidad = 0,5^(antigüedad en días / vida media)
- **Bits por peso** = bits + 16 / tamaño de grupo (4 bits, grupo 32 → 4,5)
- **Pesos (bytes)** = parámetros × bits por peso / 8
- **Caché KV (bytes)** = 2 × capas × cabezas KV × dimensión de cabeza × bytes por valor × *tokens* × lote
- **Memoria total** = (pesos + caché KV) × (1 + margen); planifica con el 75 % de la RAM
- **Techo de decodificación (una persona)** = ancho de banda de memoria / bytes de los pesos
- **Cuantización simétrica:** escala = max|w| / (2^(bits−1) − 1); código = clip(round(w / escala))
- **Latencia de extremo a extremo** = TTFT + TPOT × (*tokens* de salida − 1)
- **Decodificación especulativa:** *tokens* por pasada = (1 − α^(k+1)) / (1 − α); aceleración = *tokens* / (k·c + 1)
- **Coste de la API al día** = *tokens* de entrada/1M × precio de entrada + *tokens* de salida/1M × precio de salida
- **Coste local al día** = *hardware* / (meses × 30) + kW × horas de uso × precio por kWh
- **Servidores** = techo(*tokens*/s en el pico × (1 + margen) / *tokens*/s por servidor)

## Números del lab (según nuestra fórmula o el simulador)

| Elemento | Valor |
|---|---|
| 8B parámetros, 4 bits g32, 8.192 *tokens* (32 capas, 8 cabezas KV, dimensión 128) | 4,50 + 1,07 + 0,56 = 6,13 GB |
| Lo mismo a 16 bits | 18,78 GB (no cabe en 12 GB) |
| Caché KV por *token* (esa forma) | 128 KiB |
| Error de salida int4 por tensor frente a grupo de 32 | 0,92 frente a 0,11 |
| p50 del TTFT: *batching* estático frente a continuo (lote de 16, simulado) | 10.453 ms frente a 98 ms |
| Desperdicio de KV: contigua frente a paginada (páginas de 16 *tokens*) | 89,5 % frente a 0,9 % |
| Ablación que filtra la inyección y los PII | no-credibility + no-sanitiser: 2 trampas, 2 PII |

## Tabla de decisión: ¿qué modelo en un portátil?

| Pregunta | Si la respuesta es sí | Si la respuesta es no |
|---|---|---|
| ¿Caben pesos + KV + margen en el 75 % de la RAM? | Siguiente pregunta | Menos bits, un modelo más pequeño o un contexto más corto |
| ¿Es el techo de decodificación lo bastante rápido para la tarea? | Siguiente pregunta | Menos bits o un modelo más pequeño |
| ¿Supera tus calificaciones de la tarea (`uv run lab grade`)? | Lanza la opción más pequeña que aprueba | Modelo más grande o más bits |

## Vocabulario de servicio

| Término | Significado |
|---|---|
| TTFT | Tiempo hasta el primer *token*; incluye el tiempo en cola |
| TPOT | Tiempo por *token* de salida después del primero |
| *Batching* continuo | Las peticiones entran y salen del lote en cada paso de decodificación |
| PagedAttention | Caché KV guardada en bloques pequeños de tamaño fijo, asignados mediante una tabla de bloques |
| Decodificación especulativa | Un modelo borrador propone k *tokens*; el objetivo los verifica en una pasada |

## Lista de comprobación para un endpoint autoalojado

- Ligarlo a localhost o ponerlo detrás de una pasarela; autenticar cada petición; claves por equipo y límites de peticiones.
- TLS para todo lo que salga de la máquina; fijar los archivos de modelo por *checksum*; herramientas mínimas para el agente.
- Registrar metadatos, no *prompts*; vigilar el p95 de TTFT y TPOT, la profundidad de la cola, el uso de la caché KV, los *tokens*/s y los errores.

## Comandos del lab

```bash
uv run lab research rq1 --trace        # one report + trace -> outputs/rq1.md
uv run lab grade                        # all four questions
uv run lab grade-file rq1 outputs/rq1-edited.md
uv run lab ablate                       # defences off one at a time
uv run lab search "query" -k 5          # BM25 + credibility reasons
uv run lab memory --ram 16 --bandwidth 100
uv run lab memory --sweep --ram 16
uv run lab quantize
uv run lab serve-sim && uv run lab kv-cache && uv run lab speculate
uv run lab cost --peak-tokens-s 400
```

## Los tres errores más comunes

1. **Fiarse de un informe que parece limpio.** Califícalo: cubierto no es respaldado, y dos fuentes no son cuatro.
2. **Dimensionar un modelo por el tamaño de su archivo.** Suma la caché KV (contexto × lote) y el margen, y deja holgura.
3. **Tratar "interno" o "local" como seguridad.** Lo local es una configuración; el *endpoint* sigue necesitando autenticación, y las búsquedas siguen saliendo de la máquina.
