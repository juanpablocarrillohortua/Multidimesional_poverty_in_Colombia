<div style="width:100%; background-color:#00522c; border-bottom:3px solid #34a853; padding:20px 0; text-align:center;">
  <img src="https://pattern-lab-externado-prod.web.app/images/logo-uec.svg" alt="Banner del proyecto" width="40%">
</div>

# Columnas esperadas

El pipeline recibe una sola fila (un hogar) con las siguientes 7 columnas. La app construye esta fila a partir del formulario con `HouseHold.create_input_row()` (ver `preprocessing.py`), por lo que el usuario nunca la escribe a mano.

| # | Columna | Tipo | Valores | Variable ECV | Descripción |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `ORG_analfabetismo` | Proporción | $[0, 1]$ o vacío | `P6160` | Proporción de personas de 15 años o más del hogar que no saben leer ni escribir. |
| 2 | `ORG_trabajo_infantil` | Proporción | $[0, 1]$ o vacío | `P6240`, `P6250`, `P6260`, `P6270` | Proporción de personas de 12 a 17 años del hogar que trabajan. |
| 3 | `ORG_sin_aseguramiento` | Proporción | $[0, 1]$ | `P6090` | Proporción de personas de 6 años o más del hogar sin afiliación a salud. |
| 4 | `ORG_acueducto` | Binaria | $\{1, 2\}$ | `P8520S5` | ¿La vivienda tiene acueducto? |
| 5 | `ORG_fuente_agua_cocinar` | Ordinal | $1$ a $10$ | `P8530` (recodificada) | Fuente del agua para cocinar, de mejor a peor. |
| 6 | `ORG_alcantarillado` | Binaria | $\{1, 2\}$ | `P8520S3` | ¿La vivienda tiene alcantarillado? |
| 7 | `ORG_servicio_sanitario` | Ordinal | $1$ a $6$ | `P8526` | Tipo de servicio sanitario, de mejor a peor. |

> Nota: las columnas que pueden quedar vacías (NaN) son las que no aplican al hogar, `ORG_analfabetismo` cuando no hay personas de 15 años o más y `ORG_trabajo_infantil` cuando no hay personas de 12 a 17 años, el pipeline se encarga de imputarlas por lo que no es necesario llenarlas.

# Unidades de cada columna y variable procesada

Las proporciones se calculan sobre los miembros del hogar a los que aplica cada indicador del IPM (no sobre el total de miembros), las binarias conservan el código de la ECV y las ordinales van de la mejor a la peor categoría. Dentro del pipeline las columnas se imputan y luego se escalan con MinMaxScaler, de ahí sale la versión procesada que recibe la regresión logística:

| Columna | Unidad | Denominador / codificación | Variable procesada (entrada del modelo) |
| :--- | :--- | :--- | :--- |
| `ORG_analfabetismo` | Proporción de personas | Personas de 15 años o más | Vacío → 0, MinMax la deja igual ($[0, 1]$) |
| `ORG_trabajo_infantil` | Proporción de personas | Personas de 12 a 17 años | Vacío → 0, MinMax la deja igual ($[0, 1]$) |
| `missingindicator_ORG_trabajo_infantil` | Indicador | 1 = el hogar no tiene personas de 12 a 17 años, 0 = sí tiene | La crea el imputer, MinMax la deja igual ($\{0, 1\}$) |
| `ORG_sin_aseguramiento` | Proporción de personas | Personas de 6 años o más | Pasa directo, MinMax la deja igual ($[0, 1]$) |
| `ORG_acueducto` | Código | 1 = sí, 2 = no | MinMax: 1 → 0, 2 → 1 |
| `ORG_alcantarillado` | Código | 1 = sí, 2 = no | MinMax: 1 → 0, 2 → 1 |
| `ORG_servicio_sanitario` | Categoría ordinal | 1 (mejor) a 6 (peor) | MinMax: $(x - 1) / 5$, cada categoría vale $0.2$ |
| `ORG_fuente_agua_cocinar` | Categoría ordinal | 1 (mejor) a 10 (peor) | MinMax: $(x - 1) / 9$, cada categoría vale $\approx 0.111$ |

Note que en las proporciones MinMax se comporta como una transformación identidad (su mínimo y máximo observados son 0 y 1), por lo que un cambio de $0.1$ equivale a 10 puntos porcentuales, mientras que en las ordinales un paso de categoría equivale a $1/\text{rango}$ en la escala del modelo.

Códigos de las variables ordinales (los mismos que muestra la app en los menús desplegables):

| Código | Servicio sanitario |
| :--- | :--- |
| 1 | Inodoro conectado a alcantarillado |
| 2 | Inodoro conectado a pozo séptico |
| 3 | Inodoro sin conexión |
| 4 | Letrina |
| 5 | Inodoro con descarga directa a fuentes de agua (bajamar) |
| 6 | No tiene servicio sanitario |

| Código | Fuente del agua para cocinar |
| :--- | :--- |
| 1 | Acueducto público |
| 2 | Acueducto comunal o veredal |
| 3 | Agua embotellada o en bolsa |
| 4 | Pozo con bomba |
| 5 | Pila pública |
| 6 | Carrotanque |
| 7 | Aguatero |
| 8 | Pozo sin bomba, aljibe, jagüey o barreno |
| 9 | Agua lluvia |
| 10 | Río, quebrada, manantial o nacimiento |

> Nota: la fuente de agua está recodificada en orden de mejor a peor, no corresponde al código original del DANE de la pregunta `P8530`.

# Datos con los que se entrenó

El modelo fue entrenado con los datos recolectados por el DANE para la encuesta ECV del 2025. Solo se consideran los hogares de las cabeceras municipales (zonas urbanas), en total 44,351 hogares, y el objetivo (`POBRE`) se construyó replicando el IPM oficial del DANE con el método de doble corte de Alkire-Foster, donde aproximadamente el $12.8\%$ de los hogares resulta pobre multidimensional. Para seleccionar el modelo se dividieron los hogares en entrenamiento, validación y prueba (31,039 / 6,643 / 6,669, agrupados por segmento), y una vez evaluado en prueba el modelo de producción se reentrenó con todos los hogares.

> Nota: el umbral de decisión es de $0.45$ y se fijó en la validación cruzada para priorizar el recall, es decir, detectar la mayor cantidad posible de hogares pobres a costa de algunas falsas alarmas.


# Ejecutar con Docker

### Opción 1: Descargar desde Docker Hub (Recomendado)

La imagen es multiplataforma (`linux/amd64` y `linux/arm64`), por lo que funciona en Mac (Apple Silicon e Intel), Windows y Linux, Docker descarga automáticamente la versión de su máquina.

```bash
docker pull juanpablocarrillo1/multi-dimensional-poverty:latest
docker run --rm -p 8501:8501 juanpablocarrillo1/multi-dimensional-poverty:latest
```

### Opción 2: Construir localmente

Desde la raíz del repositorio:

```bash
docker build -t pobreza-ipm .
docker run --rm -p 8501:8501 pobreza-ipm
```

En ambos casos, la app queda disponible en <http://localhost:8501>.

# Ejecutar localmente

Desde la raíz del repositorio:

```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

la app queda disponible en <http://localhost:8501>.
