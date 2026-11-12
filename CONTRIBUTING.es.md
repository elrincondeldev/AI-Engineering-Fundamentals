🌐 [English](CONTRIBUTING.md) · Español

# Cómo contribuir

Gracias por ayudar a mejorar este curso. Son bienvenidas las correcciones de erratas, las
explicaciones más claras, los mejores ejercicios, los informes de errores en los labs y las mejoras de
traducción.

## Reglas básicas

- **Amabilidad y crítica constructiva.** Presupón buena intención y explica el porqué de cada
  sugerencia.
- **Solo trabajo original.** No pegues textos, código ni figuras de otros cursos, libros o material de
  pago. Enlaza fuentes gratuitas.
- **Nada de datos inventados.** Las estadísticas, las citas y las afirmaciones sobre modelos,
  artículos o empresas reales necesitan una fuente pública. Los datos ficticios de Hearthwise salen
  solo de [`datasets/`](datasets/README.es.md).
- **El inglés es la fuente de verdad.** Cambia primero la página en inglés, verifícala y después
  actualiza la página en español en el mismo *pull request*.
- Las contribuciones se publican bajo la [licencia MIT](LICENSE).

El contrato completo para quien escribe está en
[docs/course-conventions.es.md](docs/course-conventions.es.md): historia y personajes, nombres de
archivo, plantillas de lección y de lab, estilo de escritura, fuentes y reglas de idioma.

## Cómo proponer un cambio

1. Abre una *issue* para cualquier cosa mayor que un arreglo pequeño, así acordamos el enfoque.
2. Haz un *fork* del repositorio y crea una rama: `fix/module-05-chunk-overlap`,
   `feat/module-08-new-exercise` o `i18n/module-03-es`.
3. Haz el cambio. Mantén los *pull requests* pequeños y centrados en un módulo o en un tema.
4. Ejecuta las comprobaciones de abajo y pega su resumen en el *pull request*.
5. Describe **qué** cambiaste, **por qué** y **cómo lo verificaste**.

## Comprobaciones en local

```bash
# El lab que has tocado (cada lab es un proyecto uv independiente)
(cd modules/05-rag-retrieval/lab && uv sync && uv run pytest)

# Comprobaciones de contenido (solo biblioteca estándar)
python3 datasets/generate.py --check
python3 tools/check_links.py
python3 tools/check_i18n.py
python3 tools/check_navigation.py
python3 tools/check_lessons.py
python3 tools/check_lessons.py --lang es

# Diagramas (necesita Node.js 20 o superior)
npm ci --prefix tools
node tools/check_mermaid.mjs
```

CI las ejecuta todas en cada *pull request*. `python3 tools/check_links.py --external` comprueba
además las URL externas; ejecútalo cuando añadas enlaces.

## Cambiar una lección

- Respeta el orden de la plantilla y los emojis: las comprobaciones dependen de ellos.
- Las lecciones tienen entre 1.800 y 3.500 palabras de texto.
- **Cada número que cites debe salir de una ejecución real del lab** y comprobarse en el
  `tests/test_documented_numbers.py` de ese lab. Si cambias el código del lab y un número se mueve,
  actualiza a la vez el test, la página en inglés y la página en español.

## Cambiar un lab

- Un lab nunca importa código de otro módulo; copia lo que necesites.
- Los tests deben pasar sin conexión con el proveedor falso en menos de un minuto en la CPU de un
  portátil.
- Añade dependencias solo cuando la biblioteca estándar o `numpy` no basten, y actualiza `uv.lock`
  con `uv lock`.
- Los archivos generados van a `lab/outputs/`, que Git ignora.

## Cambiar los datasets

Edita el generador en `datasets/_gen/`, ejecuta `python3 datasets/generate.py` y después
`python3 datasets/baselines.py`. Actualiza los números de `datasets/README.md` (y de su versión en
español) y ejecuta los tests de todos los labs, porque muchos citan resultados de los datasets.

## Traducir

- Traduce a partir de la página en inglés ya verificada; mantén uno a uno los encabezados, los bloques
  de código, los bloques mermaid y los bloques `<details>` (`tools/check_i18n.py` comprueba los
  recuentos).
- El código, los identificadores y la salida de los programas se quedan en inglés; el texto y las
  etiquetas de los diagramas se traducen.
- Usa español neutro con tuteo, formas neutras en cuanto al género para los personajes y para quien
  aprende, y el término inglés en cursiva y entre paréntesis la primera vez que aparece un término
  técnico.

## Informar de un problema

Abre una *issue* indicando el módulo y la lección, qué esperabas, qué pasó y, en los labs, el comando,
la salida completa y tu sistema operativo.
