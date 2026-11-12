🌐 [English](runbook.md) · Español

# Manual de operación (*runbook*): <nombre del servicio>

> Un *runbook* le dice a la persona de guardia qué hacer, paso a paso, cuando algo falla. Escríbelo
> para una persona agotada a las 3 de la madrugada que nunca ha visto el servicio: frases cortas, comandos
> exactos, enlaces.

| Campo | Valor |
|---|---|
| Servicio | |
| Equipo responsable y turnos de guardia | |
| Paneles | |
| Logs y trazas | |
| Dependencias | *proveedor de modelos, índice vectorial, herramientas, bases de datos* |
| Última revisión | `<AAAA-MM-DD>` |

## Visión general del servicio

*Qué hace, quién depende de él y los rangos normales de las métricas clave (peticiones por minuto,
latencia p95, tasa de errores, coste por hora, tasa de abstención).*

## Alertas y primeras acciones

| Alerta | Causas probables | Primeras acciones | Escalar a |
|---|---|---|---|
| Tasa de errores por encima del umbral | Caída del proveedor, clave caducada, cambio de esquema | Revisa el estado del proveedor; cambia al proveedor de respaldo | |
| Latencia p95 por encima del umbral | Proveedor lento, contextos largos, reintentos | Revisa el número de *tokens* y la tasa de reintentos; reduce el presupuesto de contexto | |
| Coste por hora por encima del presupuesto | Crecimiento del *prompt*, tormenta de reintentos, abuso | Revisa los usuarios con más consumo y el tamaño de los *prompts*; activa límites de uso | |
| Pico de respuestas "no lo sé" | Índice sin actualizar, recuperación rota | Ejecuta la prueba rápida de recuperación; revisa la última construcción del índice | |
| Salta el detector de inyección o de fuga de datos | Contenido malicioso en fuentes o entradas | Bloquea la fuente; conserva las trazas; avisa a Seguridad | |

## Procedimientos habituales

### Cambiar al proveedor de modelos de respaldo

1. *Comando exacto o cambio de configuración*
2. *Cómo verificar que funcionó*

### Volver a una versión anterior del prompt o del modelo

1. *Pasos exactos*
2. *Cómo verificarlo*

### Reconstruir el índice de recuperación

1. *Pasos exactos*
2. *Cómo verificarlo (preguntas de prueba y fuentes esperadas)*

## Verificación tras cualquier cambio

- [ ] Las pruebas rápidas pasan
- [ ] La batería de evaluación alcanza el umbral de lanzamiento
- [ ] Los paneles vuelven a sus rangos normales

## Contactos

| Rol | Nombre | Cómo contactar |
|---|---|---|
| | | |

---

[🏠 Inicio del curso](../README.es.md) · [Plantilla de postmortem →](postmortem.es.md)
