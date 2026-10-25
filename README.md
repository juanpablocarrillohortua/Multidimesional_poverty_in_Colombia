<div style="width:100%; background-color:#00522c; border-bottom:3px solid #34a853; padding:20px 0; text-align:center;">
  <img src="https://pattern-lab-externado-prod.web.app/images/logo-uec.svg" alt="Banner del proyecto" width="40%">
</div>

# Pobreza multidimensional en Colombia

[![quality](https://github.com/juanpablocarrillohortua/Multidimesional_poverty_in_Colombia/actions/workflows/quality.yml/badge.svg)](https://github.com/juanpablocarrillohortua/Multidimesional_poverty_in_Colombia/actions/workflows/quality.yml)
![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)

Proyecto del curso de Machine Learning (Universidad Externado de Colombia) que
predice si un hogar urbano está en pobreza multidimensional (`POBRE`) a partir
de los microdatos de la Encuesta Nacional de Calidad de Vida (ECV) 2025 del
DANE. El objetivo se construye con el doble corte de Alkire-Foster, replicando
el Índice de Pobreza Multidimensional (IPM) oficial.

## Orden de ejecución

1. Cree la carpeta `data/` en la raíz del repositorio.
2. Descargue la carpeta ECV 2025 desde
   [Google Drive](https://drive.google.com/drive/folders/14QHGdfYVWSl_d4xQcaucNlVdXCh8a8ZU)
   y guárdela como `data/ECV_2025`.
3. Cree un entorno virtual e instale las dependencias de `requirements.txt`
   (`make install` o `pip install -r requirements.txt`).
4. Descomprima la encuesta:

   ```bash
   python domain_ultis/unzip_year.py data/ECV_2025
   ```

5. Ejecute completo `notebooks/eda_consolidate.ipynb`. Al final exporta los
   conjuntos de train, validación y test a `data/splits/`.
6. Ejecute `notebooks/modeling.ipynb`, que lee esos conjuntos.

## Orden de lectura sugerido

1. `notebooks/full_analisis.ipynb`: muestra cómo cruzar las tablas de la ECV
   (creado con Claude).
2. `notebooks/eda_consolidate.ipynb`: análisis exploratorio de la base
   consolidada y justificación de las transformaciones que se usan en el
   modelado.
3. `notebooks/modeling.ipynb` y `notebooks/modelling2.ipynb`: propuestas de modelos, evaluación con validación
   cruzada y comparación global.


## Estructura del repositorio

```text
Multidimesional_poverty_in_Colombia/
├── .github/workflows/quality.yml  # CI: make quality en cada push y PR a main
├── config.py                      # rutas del proyecto (WORKDIR, DATADIR, EXTRACTED)
├── data/                          # no versionada: ECV_2025/, extracted/, splits/, cache_cv/
├── docs/                          # artículo de referencia (Zhao, Cheah y Moses, 2024) e informe de la entrega
├── domain/
│   ├── ipm.py                     # transformer IPM: doble corte Alkire-Foster → POBRE
│   └── variables_naming.py        # códigos de la ECV → pregunta, archivo y nivel
├── domain_ultis/
│   └── unzip_year.py              # descomprime los zips de un año de la ECV
├── notebooks/
│   ├── full_analisis.ipynb        # cómo cruzar las tablas de la ECV
│   ├── consolidate_dfs.ipynb      # exploración inicial de las llaves de cruce
│   ├── eda_consolidate.ipynb      # consolidación, objetivo, partición, EDA y exportación
│   └── modeling.ipynb             # propuestas de modelos y comparación global
├── outputs/
│   ├── plots/                     # figuras generadas
│   └── tables/                    # tablas generadas
├── tools/                         # clean.py y mk_help.py (soporte del Makefile)
├── utils/                         # módulos de análisis y EvaluadorCV (ver utils/README.md)
├── Makefile                       # install, lint, format, quality
├── requirements.txt               # dependencias de ejecución
├── requirements/dev.txt           # ruff (linter y formateador)
├── ruff.toml                      # reglas de estilo: 79 columnas, E/W/F/I/N
├── references.bib                 # referencias en BibTeX
├── CLAUDE.md                      # guía del repositorio para Claude Code
└── LICENSE
```

## Reconocimiento a Claude Opus 5.5

Este proyecto se desarrolló con el apoyo de Claude Opus 5.5 (Anthropic), usado
como asistente de programación a través de Claude Code. Claude ayudó a escribir
el cuaderno `full_analisis.ipynb` sobre cómo cruzar las tablas de la ECV y la
guía `CLAUDE.md`; a implementar la partición en train, validación y test
agrupada por `SEGMENTO` y estratificada por `POBRE`; a escribir módulos de
`utils/` como `cat_cat_dep.py`, la versión ponderada de `join_plots.py` y la
clase `EvaluadorCV` (validación cruzada agrupada, caché, AOR, pseudo-R² e
intervalos de confianza por bootstrap de conglomerados); a revisar posibles
fugas de información en los pipelines; a construir la comparación global de
modelos; y a mantener el código dentro de las reglas de estilo
(`make quality`). Las decisiones de modelado, la elección de los umbrales y del
modelo ganador, y las interpretaciones corresponden al equipo, que revisó y
ejecutó el código.

## Referencias

- Alkire, S., & Foster, J. (2011). Counting and multidimensional poverty
  measurement. *Journal of Public Economics, 95*(7–8), 476–487.
  https://doi.org/10.1016/j.jpubeco.2010.11.006
- Departamento Administrativo Nacional de Estadística (DANE). (2022).
  *Metodología del Índice de Pobreza Multidimensional*.
  https://www.dane.gov.co/files/operaciones/PM/met-IPM.pdf
- Departamento Administrativo Nacional de Estadística (DANE). (2025).
  *Encuesta Nacional de Calidad de Vida (ECV) 2025*.
  https://microdatos.dane.gov.co/index.php/catalog/905/study-description
- Departamento Administrativo Nacional de Estadística (DANE). (2025).
  *Índice de Pobreza Multidimensional (IPM)*.
  https://microdatos.dane.gov.co/index.php/catalog/903/study-description
- Departamento Administrativo Nacional de Estadística (DANE). (2025).
  *Materiales relacionados con la Encuesta Nacional de Calidad de Vida*.
  https://microdatos.dane.gov.co/index.php/catalog/905/related-materials
- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array
  programming with NumPy. *Nature, 585*, 357–362.
  https://doi.org/10.1038/s41586-020-2649-2
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in
  Science & Engineering, 9*(3), 90–95. https://doi.org/10.1109/MCSE.2007.55
- Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A
  Python toolbox to tackle the curse of imbalanced datasets in machine
  learning. *Journal of Machine Learning Research, 18*(17), 1–5.
  https://jmlr.org/papers/v18/16-365.html
- McKinney, W. (2010). Data structures for statistical computing in Python. En
  *Proceedings of the 9th Python in Science Conference* (pp. 56–61).
  https://doi.org/10.25080/Majora-92bf1922-00a
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn:
  Machine learning in Python. *Journal of Machine Learning Research, 12*(85),
  2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
- Python Software Foundation. (2025). *Python documentation*.
  https://docs.python.org/3/
- scikit-learn developers. (2025). *Cross-validation: Evaluating estimator
  performance*. https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn developers. (2025). *Pipeline*.
  https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical
  modeling with Python. En *Proceedings of the 9th Python in Science
  Conference*. https://conference.scipy.org/proceedings/scipy2010/seabold.html
- Streamlit. (2025). *Streamlit documentation*. https://docs.streamlit.io/
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0:
  Fundamental algorithms for scientific computing in Python. *Nature Methods,
  17*, 261–272. https://doi.org/10.1038/s41592-019-0686-2
- Waskom, M. L. (2021). Seaborn: Statistical data visualization. *Journal of
  Open Source Software, 6*(60), 3021. https://doi.org/10.21105/joss.03021
- Zhao, Cheah, & Moses. (2024). Measurement and determinants of
  multidimensional urban poverty: Evidence from Shandong Province, China.
  *PLOS ONE, 19*(5), e0300263. https://doi.org/10.1371/journal.pone.0300263
