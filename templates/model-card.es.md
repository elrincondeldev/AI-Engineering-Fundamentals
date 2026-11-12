🌐 [English](model-card.md) · Español

# Ficha del modelo (*model card*): <nombre del modelo o del sistema>

> Una ficha de modelo documenta para qué sirve un modelo o un sistema con LLM, cómo se evaluó y dónde
> falla, siguiendo la idea de [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993).
> Rellénala para cada modelo o sistema *prompt*+modelo que publiques, incluidos los adaptadores
> afinados.

## Detalles del modelo

| Campo | Valor |
|---|---|
| Nombre y versión | |
| Tipo | *por ejemplo: adaptador LoRA sobre un modelo de pesos abiertos, sistema RAG sobre una API de modelos* |
| Modelo base y proveedor | |
| Responsable y contacto | |
| Fecha | |
| Licencia | |

## Uso previsto

- **Uso principal:** *quién lo usa y para qué*
- **Usuarios:** *clientela, equipo de soporte, analistas...*
- **Fuera de alcance:** *usos para los que no debe emplearse (por ejemplo decisiones médicas, legales o de pago)*

## Datos

| Datos | Origen | Tamaño | Notas (licencia, datos personales, sesgos conocidos) |
|---|---|---|---|
| Entrenamiento / adaptación | | | |
| Corpus de recuperación | | | |
| Evaluación | | | |

## Evaluación

| Métrica | Dataset / segmento | Resultado | Umbral de lanzamiento |
|---|---|---|---|
| | Global | | |
| | *Inglés frente a español* | | |
| | *Preguntas sin respuesta* | | |

*Indica cómo se obtuvieron los resultados (comando,* commit*, proveedor y modelo) para que cualquiera
pueda reproducirlos.*

## Limitaciones y modos de fallo conocidos

- *Por ejemplo: los datos documentados solo en español a menudo se pierden en preguntas en inglés.*

## Seguridad, privacidad y ética

- **Datos personales:** *qué se anonimiza, qué se guarda y durante cuánto tiempo*
- **Inyección de prompts y mal uso:** *mitigaciones y riesgo residual*
- **Equidad:** *segmentos revisados, diferencias encontradas*

## Monitorización y mantenimiento

*Qué se monitoriza en producción, quién lo revisa y qué provoca una nueva evaluación o una vuelta
atrás.*

---

[🏠 Inicio del curso](../README.es.md) · [Plantilla de runbook →](runbook.es.md)
