🌐 [English](README.md) · Español

# Módulo 04 — Adaptar LLM: *prompting*, contexto y *fine-tuning*

> Haz que un LLM de propósito general sea bueno en un trabajo concreto, enrutar tickets de soporte,
> cambiando su *prompt*, su contexto o sus pesos, y demuestra cada paso con una evaluación que no se
> pueda engañar a sí misma.

## 🗺️ Mapa del curso

```mermaid
flowchart LR
    subgraph P1["Parte 1 · Fundamentos"]
        M01["01 · Cómo se construyen los LLM"] --> M02["02 · Del modelo base al asistente"] --> M03["03 · Playground de LLM"]
    end
    subgraph P2["Parte 2 · Adaptación y RAG"]
        M04["04 · Prompting y fine-tuning"] --> M05["05 · RAG I: recuperación"] --> M06["06 · RAG II: bot de soporte"]
    end
    subgraph P3["Parte 3 · Agentes"]
        M07["07 · Workflows y herramientas"] --> M08["08 · Agente Ask-the-Web"]
    end
    subgraph P4["Parte 4 · Razonamiento"]
        M09["09 · Modelos de razonamiento"] --> M10["10 · Deep research y local"]
    end
    M03 --> M04
    M06 --> M07
    M08 --> M09
    M10 --> CAP["🎓 Proyecto final"]
    classDef current fill:#ffb300,stroke:#e65100,stroke-width:3px,color:#000
    class M04 current
```

## 📖 La historia

El *playground* que construiste en el módulo 03 ya está en marcha, y todos los equipos de Hearthwise
prueban *prompts* en él. Alex Moreau, responsable de atención al cliente, tiene una petición concreta:
cada ticket nuevo debería llegar solo a la cola correcta, entre ocho intenciones, en inglés y en
español. Hoy, alguien del equipo de soporte pasa el primer minuto de cada ticket solo decidiendo
adónde va.

Dani Ruiz, responsable de evaluación, está de acuerdo con una condición: ninguna afirmación sin una
evaluación honesta. Dani sabe que los datos de los tickets esconden trampas: una columna que delata la
respuesta, personas que envían el mismo ticket dos veces, alrededor de un 4 % de etiquetas erróneas y
números de teléfono en las firmas. Y el 2025-09-01 Hearthwise lanzó la cámara Lookout C2, así que los
tickets posteriores a esa fecha hablan de cosas que los anteriores nunca mencionan.

En este módulo llevas el enrutador de un *prompt* desnudo a un contexto con presupuesto y compactado,
y después decides con números si merece la pena hacer *fine-tuning* de un adaptador LoRA. La pregunta
de Rio para esta semana es la que recorre todas las lecciones: ¿qué medirías antes de fiarte?

## 🎯 Objetivos de aprendizaje

Al terminar este módulo puedes:

- Elegir entre *prompting*, ingeniería de contexto, recuperación y *fine-tuning* con un marco basado en
  los datos, el coste, la latencia, la frescura y el control.
- Construir una evaluación honesta para una funcionalidad con LLM: partición temporal, deduplicación,
  auditoría de fugas, métricas de deriva y exactitud por segmento.
- Escribir y comparar *prompts* sin ejemplos (*zero-shot*) y con pocos ejemplos (*few-shot*)
  (selección, equilibrio y orden de los ejemplos), con cadena de pensamiento, con rol y con contexto de
  la persona usuaria, con salida JSON validada.
- Ensamblar el contexto con un presupuesto de *tokens*, ordenarlo de forma deliberada, compactarlo,
  aprovechar la caché de su prefijo estable e informar de lo que se descartó.
- Explicar e implementar LoRA (`W + (alpha/r)·B·A`), calcular su número de parámetros, compararla con
  el *fine-tuning* completo y con QLoRA, y detectar el olvido catastrófico.

## 🗂️ Lecciones

| # | Lección | Tiempo | Qué aprenderás |
|---|---|---|---|
| 4.1 | [Elegir cómo adaptar un LLM y medirlo con honestidad](01-adaptation-techniques.es.md) | 50 min | El panorama de la adaptación, un marco de decisión, particiones temporales, fuga del objetivo y deriva |
| 4.2 | [Ingeniería de *prompts* para el enrutador de intenciones](02-prompt-engineering.es.md) | 55 min | *Prompts* sin ejemplos y con pocos ejemplos, selección de ejemplos, cadena de pensamiento, rol y contexto de la persona usuaria, validación JSON y reintentos |
| 4.3 | [Ingeniería de contexto: ensamblar, presupuestar y compactar](03-context-engineering.es.md) | 50 min | Piezas del contexto y prioridades, presupuestos de *tokens*, orden, compactación, caché de prefijo y degradación del contexto |
| 4.4 | [*Fine-tuning*, PEFT y LoRA](04-fine-tuning-peft-and-lora.es.md) | 60 min | *Fine-tuning* completo frente a PEFT, las matemáticas de LoRA, QLoRA, cuándo el *fine-tuning* gana al *prompting*, olvido |
| Lab | [Un enrutador de intenciones, adaptado de tres formas](lab/README.es.md) | 2,5 h | Construye la auditoría, el arnés, el constructor de *prompts*, el ensamblador de contexto y LoRA en numpy |

## 🧪 Lab

Construyes un enrutador de intenciones para 1.224 tickets de soporte de Hearthwise y lo adaptas de
tres formas: con *prompting*, con ingeniería de su contexto y con *fine-tuning* de un clasificador
congelado mediante LoRA escrito desde cero en numpy. Un aprendiz en contexto de juguete, transparente,
ocupa el lugar del LLM, así que todo funciona sin conexión, de forma determinista y en segundos; el
mismo arnés puede llamar a un modelo local a través de Ollama o a cualquier *endpoint* compatible con
OpenAI.

```bash
cd modules/04-prompting-context-and-fine-tuning/lab
uv sync
uv run pytest
uv run lab --help
```

Instrucciones completas: [lab/README.es.md](lab/README.es.md).

Definición de terminado (*Definition of Done*):

- [ ] Todos los tests pasan, incluidos los cuatro ejercicios que implementas (`exercises.py`).
- [ ] Sabes explicar por qué el mismo *prompt* obtiene 0.959 con una partición aleatoria y 0.828 con
      una temporal.
- [ ] Encontraste la columna con fuga y sabes enumerar todos los caminos por los que podría entrar en
      un *prompt*.
- [ ] Reprodujiste la tabla de comparación de *prompts* y elegiste una variante con su coste en
      *tokens*.
- [ ] Explicaste un informe de descartes del ensamblador de contexto.
- [ ] Comparaste la adaptación por el lado del *prompt* con LoRA en tickets nuevos y antiguos, y
      escribiste una recomendación.

## 📏 Rúbrica de autoevaluación

| Criterio | 1 Inicial | 2 En desarrollo | 3 Competente | 4 Avanzado |
|---|---|---|---|---|
| Elegir una técnica de adaptación | Nombra el *prompting* y el *fine-tuning* | Describe las cinco técnicas y qué cambia cada una | Aplica el marco de datos/coste/latencia/frescura/control a una tarea nueva y justifica el orden de los intentos | Diseña un plan por etapas con compuertas medibles y un respaldo, y lo defiende ante una revisión escéptica |
| Evaluación honesta | Sabe que el conjunto de prueba debe estar separado | Ejecuta la partición temporal del lab | Encuentra duplicados, fugas y deriva en un dataset desconocido y explica su efecto en la puntuación | Construye una evaluación con compuertas de datos antiguos, segmentos y monitorización, y detecta un resultado defectuoso antes del lanzamiento |
| Ingeniería de *prompts* | Sabe escribir un *prompt* sin ejemplos | Compara las variantes de *prompt* del lab | Elige ejemplos, definiciones y contexto para una tarea nueva y valida la salida estructurada | Hace experimentos controlados de *prompts* con un modelo real, con su coste, y los convierte en una versión de *prompt* versionada |
| Ingeniería de contexto | Sabe que la ventana de contexto es finita | Usa el ensamblador con un presupuesto | Fija prioridades, orden y compactación para una mezcla nueva de piezas y explica el informe de descartes | Diseña presupuestos por pieza, prefijos aptos para la caché y monitorización para un sistema en producción |
| *Fine-tuning* y LoRA | Sabe que el *fine-tuning* cambia los pesos | Calcula el número de parámetros de LoRA | Explica la actualización de LoRA y sus gradientes, y compara LoRA con el *fine-tuning* completo en datos nuevos y antiguos | Decide cuándo compensa un adaptador, planifica los datos, la repetición y la marcha atrás, y dimensiona la memoria para QLoRA |

## 🎤 Rincón de entrevistas

**1. ¿Cuándo harías *fine-tuning* en lugar de mejorar el *prompt*?**

<details><summary>Respuesta</summary>

Cuando la tarea es estable, tienes suficientes ejemplos limpios y etiquetados, y el *prompt* ha
llegado a un techo medido en calidad, coste o latencia. Señales típicas: *prompts* *few-shot* largos
que se pagan en cada petición con mucho volumen, un formato o un estilo que el *prompt* no consigue de
forma fiable, o presupuestos de latencia ajustados. No haría *fine-tuning* cuando las etiquetas o los
hechos cambian cada semana, cuando hay pocas etiquetas o antes de tener una línea base sólida de
*prompt*, porque entonces no podría saber si el entrenamiento ayudó. En el lab de este módulo, el
*fine-tuning* ganó al mejor *prompt* actualizado en los tickets nuevos (0.927 frente a 0.901), pero
solo reescribir las definiciones de las etiquetas ya subió los tickets de la cámara de 0.496 a 0.851
sin ninguna etiqueta.

</details>

**2. Explica LoRA en dos minutos, incluido el número de parámetros.**

<details><summary>Respuesta</summary>

LoRA congela una matriz de pesos preentrenada `W` (`d_out × d_in`) y aprende una actualización de
bajo rango `(alpha/r)·B·A`, con `B` de forma `d_out × r` y `A` de forma `r × d_in`. Solo se entrenan A
y B: `r·(d_in + d_out)` números en lugar de `d_in·d_out`. B empieza en cero, así que el entrenamiento
parte exactamente del modelo base. Tras el entrenamiento, la actualización se puede fusionar con W,
así que la inferencia no cuesta nada extra. Para una matriz de 4.096 × 4.096, el rango 8 entrena
65.536 números, el 0,39 % de la matriz. Solo ahorra parámetros cuando r es mucho menor que las dos
dimensiones.

</details>

**3. Tu exactitud sin conexión es del 96 %, pero la gente se queja. ¿Qué compruebas primero?**

<details><summary>Respuesta</summary>

Cómo se construyó el conjunto de prueba. Las particiones aleatorias con duplicados y sin orden
temporal inflan las puntuaciones: en este módulo, el mismo *prompt* obtuvo 0.959 con una partición
aleatoria y 0.828 con una temporal. Después compruebo si hay fuga del objetivo (un campo que solo se
conoce a posteriori), deriva desde que se recogió el conjunto de prueba y segmentos que la media
oculta, como un producto nuevo o un segundo idioma.

</details>

**4. ¿Cómo mantienes asequible un *prompt* que se usa durante mucho tiempo?**

<details><summary>Respuesta</summary>

Fija un presupuesto de *tokens* por petición y ensambla el contexto por prioridad, informa de lo que
se descarta, limpia y deduplica el texto, resume el historial antiguo y mantén la parte estable del
*prompt* al principio e idéntica byte a byte para que funcione la caché de prefijo. Después vigila los
*tokens* por petición, la proporción de peticiones que llegan al límite del presupuesto y la tasa de
aciertos de la caché. Basta con añadir un ID de petición al principio del *prompt* de sistema para que
la tasa de aciertos caiga a cero.

</details>

**5. ¿Qué es el olvido catastrófico y lo evita LoRA?**

<details><summary>Respuesta</summary>

Es la pérdida de habilidades anteriores cuando un modelo se sigue entrenando con datos nuevos y
acotados. LoRA limita cuánto puede cambiar el modelo, pero no la dirección del cambio, así que por sí
sola no evita el olvido: en el lab, LoRA de rango 4 perdió en los tickets antiguos más o menos tanta
exactitud como el *fine-tuning* completo tras un entrenamiento largo. Lo que ayuda es una compuerta de
evaluación con datos antiguos, la repetición de ejemplos antiguos, la parada temprana y la propiedad de
PEFT de que los pesos base quedan intactos, así que el adaptador se puede desactivar.

</details>

## ✅ Antes de seguir

- Haz el [quiz del módulo](quiz.es.md): 14 preguntas, tres de ellas repasan los módulos 02 y 03.
- Ten la [chuleta](cheatsheet.es.md) a mano para los próximos módulos; el módulo 05 reutiliza las
  ideas de selección de ejemplos y de contexto a la escala de todo un centro de ayuda.
- Anota tu recomendación para Alex y Dani en tu [diario de progreso](../../docs/progress-journal.es.md).

---

[← Módulo 03](../03-llm-playground/README.es.md) · [🏠 Inicio del curso](../../README.es.md) · [Empieza: Lección 4.1 →](01-adaptation-techniques.es.md)
