🌐 [English](cheatsheet.md) · Español

# Chuleta del módulo 04 — Adaptar LLM

## Términos clave

| Término | Significado |
|---|---|
| *Zero-shot* / *few-shot* | *Prompt* sin ejemplos resueltos / con k ejemplos resueltos (aprendizaje en contexto) |
| *Prompting* con rol (*role-specific prompting*) | Un mensaje de sistema que fija quién es el modelo, su alcance y sus reglas |
| *Prompting* con contexto de la persona usuaria (*user-context prompting*) | Datos de quien escribe que se conocen al predecir (idioma, producto, canal) |
| Cadena de pensamiento (CoT, *chain of thought*) | Pedir un razonamiento intermedio antes de la respuesta; para problemas de varios pasos |
| Ingeniería de contexto (*context engineering*) | Elegir qué piezas entran en la ventana, su tamaño, su orden y su prioridad |
| Compactación (*compaction*) | Limpiar, deduplicar, resumir, truncar |
| Caché de prefijo (*prefix caching*) | Reutilizar el trabajo del proveedor para un prefijo de *prompt* idéntico byte a byte |
| Degradación del contexto (*context rot*) | La calidad baja a medida que la ventana se llena de material desfasado, redundante o marginal |
| PEFT | Ajuste fino eficiente en parámetros (*parameter-efficient fine-tuning*): congelar la base y entrenar un pequeño complemento |
| LoRA / QLoRA | Actualización de bajo rango de pesos congelados / lo mismo sobre una base de 4 bits |
| Fuga del objetivo (*target leakage*) | Una variable que solo se conoce cuando la respuesta ya existe (`resolution_code`) |
| Deriva de concepto (*concept drift*) | Cambia la relación entre entradas y etiquetas (lanzamiento de la Lookout C2, 2025-09-01) |
| Olvido catastrófico (*catastrophic forgetting*) | Perder habilidades antiguas tras entrenar con datos nuevos y acotados |

## Fórmulas

```text
LoRA update        W' = W + (alpha / r) · B · A        B: d_out × r (zeros at start), A: r × d_in (random)
LoRA size          r · (d_in + d_out)                  full matrix: d_in · d_out
Share of full      r · (d_in + d_out) / (d_in · d_out)  4096 × 4096, r = 8 -> 65,536 = 0.00391
LoRA gradients     grad_W = dLoss/dW'   grad_B = (alpha/r) · grad_W · Aᵀ   grad_A = (alpha/r) · Bᵀ · grad_W
Weights memory     params × bits / 8 bytes             7e9 params: 14.0 GB at 16-bit, 3.5 GB at 4-bit
Jaccard            |A ∩ B| / |A ∪ B|
MMR score          (1 - λ) · sim(query, c) - λ · max sim(c, chosen)
```

## Elegir una técnica

| Pregunta | *Prompt* / contexto / RAG | *Fine-tuning* (PEFT) |
|---|---|---|
| ¿Datos etiquetados? | Pocos o ninguno | Cientos o más, limpios |
| ¿Dónde pagar? | Por petición (*tokens*) | Por adelantado (entrenamiento) |
| ¿Presupuesto de latencia? | Moderado | Ajustado (*prompts* más cortos) |
| ¿Cada cuánto cambia? | A menudo | Rara vez |
| ¿Qué hay que controlar? | Lógica legible y auditable | Formato o habilidad que un *prompt* no consigue mantener |

Orden de intentos: *prompt* + evaluación honesta → contexto / recuperación → LoRA → (rara vez)
*fine-tuning* completo.

## Resultados del lab que debes recordar (prueba temporal, 400 tickets)

| Resultado | Valor |
|---|---|
| Mismo *prompt*, partición aleatoria frente a temporal | 0.959 frente a 0.828 |
| `resolution_code` solo / pegado en el *prompt* | 1.000 / 0.920 |
| Solo nombres de etiquetas → definiciones bilingües (0 ejemplos) | 0.270 → 0.723 |
| 8 ejemplos aleatorios frente a 8 parecidos | 0.728 frente a 0.828 |
| + contexto de la persona usuaria (idioma, producto, canal) | 0.845, 790.5 *tokens* de *prompt* |
| Paso a paso: exactitud, *tokens* de respuesta | 0.828, 9.4 → 35.7 |
| JSON válido al primer intento (estricto / tolerante), tras 1 reintento | 379 / 390 de 400, 0 no válidas |
| Tickets con respuesta citada, 8 ejemplos en bruto frente a limpios | 0.596 frente a 0.830 |
| Tasa de aciertos de la caché de prefijo, estable frente a ID de petición al principio | 0.998 frente a 0.000 |
| Oct–dic: *prompt* actualizado / LoRA r=4 / *fine-tuning* completo | 0.901 / 0.905 / 0.927 |
| Agosto tras 4.000 pasos: completo / LoRA r=4 / completo + repetición | 0.792 / 0.812 / 0.865 |

## Lista de comprobación del ensamblado de contexto

1. Lo obligatorio primero: el *prompt* de sistema y la consulta; nunca los descartes.
2. Piezas opcionales por prioridad hasta llenar el presupuesto; registra cada descarte con su motivo.
3. Orden: prefijo estable → ejemplos (del menos al más relevante) → perfil → consulta.
4. Limpia antes que nada (citas, firmas, entidades HTML, datos personales).
5. Mide la deduplicación y el resumen; no supongas que ayudan.
6. Saca todo lo que varía (ID, fechas) de la parte de arriba del *prompt*.

## Comandos del lab

```bash
cd modules/04-prompting-context-and-fine-tuning/lab && uv sync && uv run pytest
uv run lab audit                 # duplicates, PII, leak audit, drift
uv run lab splits                # random vs temporal split, leak through the prompt
uv run lab show-prompt --k 2     # the exact messages for one ticket (--ticket T00843)
uv run lab prompts               # zero/few-shot, selection, user context
uv run lab format                # JSON validation, retries, step-by-step cost
uv run lab context --budget 500  # one assembled prompt and its drop report
uv run lab budget                # budgets x compaction policies, too many examples
uv run lab cache                 # prefix cache simulation
uv run lab drift                 # adapt by rewriting definitions or refreshing the pool
uv run lab lora                  # LoRA ranks vs full fine-tuning, forgetting, data curve
uv run lab lora-params           # LoRA size and weights memory
```

## Los tres errores más comunes

1. **Fiarte de una partición aleatoria.** Los duplicados y la ausencia de deriva inflan la
   puntuación; parte por tiempo y deduplica primero.
2. **Dejar entrar una fuga por el contexto.** Un campo escrito a posteriori puede entrar por el
   *prompt*, por el banco de ejemplos o por el índice de recuperación, no solo por las variables.
3. **Hacer *fine-tuning* sin una compuerta de datos antiguos.** Puntúa los datos nuevos *y* los
   antiguos; un rango bajo no evita el olvido, la repetición y la parada temprana sí ayudan.
